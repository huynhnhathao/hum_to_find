import logging
import json
import os

from typing import Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torchaudio

from constants import *
from inception_resnet import *



logger = logging.getLogger()


# TODO: can not predetermine the num_chunks when inference, or predetermine the num_samples
# TODO: cache the transformed audio,
# TODO: Add multiprocessing

class Evaluator:
    """
    Evaluator class evaluate the given model on the given dataset, using `l2` distance
    or `adaptive`
    """
    def __init__(self, model,
                annotation_file: str,
                audio_dir: str,
                distance_method: str,
                transformation: str,
                target_sample_rate: int, 
                singing_threshold: float,
                device: str,
                save_embeddings_path: str,
                save_features_path: str,
                save_embeddings: bool = True,
                
                ) -> None:
        """
        Args:
            model: torch model to extract embeddings
            cached_path: path to save the embedding vectors of audios
            annotation_file: path to the annotation csv
            audio_dir: dir to the audio files
            distance_method: either `l2` or `apdaptive`
            transformation: either `mel_spectrogram` or `spectrogram`
            target_sample_rate: sameple rate use to resample the wave audio
            num_samples: number of samples for each example
            singing_threshold: threshold to consider someone is singing, use to 
                cut head of the wave file
            device: cpu or cuda
            save_embedding: whether to save the embeddings to files or not
            save_embedding_path: path to save the embeddings
            save_features_path: path to folder to cache the transformed features
        """
        self.model = model
        self.annotation = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.distance_method = distance_method
        self.transformation = self._get_transformation(transformation)
        self.target_sample_rate = target_sample_rate
        self.singing_threshold = singing_threshold
        self.device = device
        self.save_embeddings_path = save_embeddings_path
        self.save_features_path = save_features_path
        self.save_embeddings = save_embeddings

        # save all song and hum embeddings to measure distances later
        self.all_song_embeddings = []
        self.all_hum_embeddings = []

    def _get_transformation(self, transformation: str) -> Any:
        transformer = None
        if transformation == 'mel_spectrogram':
            transformer = torchaudio.transforms.MelSpectrogram(
                            sample_rate = SAMPLE_RATE,
                            n_fft = TRANSFORMER_NFFT,
                            hop_length = TRANSFORMER_HOP_LENGTH,
                            n_mels = N_MELS)
        if transformer is None:
            raise ValueError('Transformer not specified')
        return transformer

    def _resample_if_necessary(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample the signal with sample rate = sr into self.target_sample_rate"""
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """mix down to one channel if there are more than one channel"""
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cut_head_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """Cut the head of the signal until someone start singing
        """
        # TODO: is this necessary?
        first_index = 0
        for value in signal[0, :]:
            first_index += 1
            if np.abs(value.item()) > self.singing_threshold:
                break

        return signal[:, first_index:]

    def _cut_tail_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _split_signal(self, signal: torch.Tensor,                               #TODO: num_chunks depend on the duration of the audio
                    chunk_len: int, overlapping: int,
                   ) -> List[torch.Tensor]:


        """plit the signal into num_chunks equal chunks, 
        each chunk overlapping of overlapping samples
        Args:
            signal: the tensor signal to be splited
            chunk_len: number of samples for each chunk
            overlapping: the number of samples overlapping between chunks

        # NOTE: right now overlapping must be a half of chunk_len,
        #  or this method will be broken
        """

        # number of samples in signal must % overlapping == 0
        num_chunks = (signal.shape[-1]//overlapping) - 1
        all_chunks = []
        for i in range(num_chunks):
            all_chunks.append(signal[:, i*overlapping:(i+2)*overlapping])
        # instead of drop the final signals that does not fit in our chunk size, 
        # we will take the last chunk_size samples as our final chunk
        all_chunks.append(signal[:, -chunk_len: ])
        return all_chunks

    def _transformation(self, signals: List[torch.Tensor]) -> List[torch.Tensor]:
        """transform each signal in signals into mel-spectrogram"""
        transformed_signals = []
        for signal in signals:
            # add batch dimension
            transformed_signals.append(self.transformation(signal).unsqueeze(0))

        return transformed_signals

    def _save_signals_features(self, signals: torch.Tensor, audio_path: str) -> None:
        """Save the tensor of signals into disk
        Args:
            signals: a tensor of shape (batch, channel, features,..)
            signal_path: path of the audio, use to infer the name for the saved file
        """

        # save the signal tensor into self.save_features_path, with the name
        # song_<filename>.pt for song audio and hum_<filename>.pt for hummed audio
        filename = '_'.join(audio_path.split('/')[-2:])
        filename = filename.split('.')[0] + '.pt'

        with open(os.path.join(self.save_features_path, filename), 'wb') as f:
            torch.save(signals.to('cpu'), f)

    def _retrieve_signals_if_exist(self, audio_path: str )-> torch.Tensor:
        """Load the signals features from the disk
        Args:
            audio_path: path to the audio that want to retrieve the saved features
                note that this is not the path to the audi features
        """
        filename = '_'.join(audio_path.split('/')[-2:])
        filename = filename.split('.')[0] + '.pt'
        filepath = os.path.join(self.save_features_path, filename)
        features = None
        if os.path.isfile(filepath):
            features = torch.load(open(filepath, 'rb'))
            return features

        return features
        

    def _preprocess_and_embed_one_audio(self,
                                        audio_path: str) -> np.ndarray:
        """Preprocess one audio, split it into chunks, transform it and forward
        it through the model
        Args:
            audio_path: relative path to the audio

        Returns: Embedding vectors for each chunk
        """
        # looking for cached features before actually compute it
        
        signals = self._retrieve_signals_if_exist(audio_path)
        if signals is not None:
            print('Retrieving cached features')
        if signals is None:
            signal, sr = torchaudio.load(audio_path)
            signal = signal.to(self.device)
            signal = self._resample_if_necessary(signal, sr)
            signal = self._mix_down_if_necessary(signal)
            # signal = self._cut_head_if_necessary(signal) No cut head when evaluating
            # signal = self._cut_tail_if_necessary(signal) No cut tail when elvaluating
            # signal = self._right_pad_if_necessary(signal) No right pad when evaluating
            signals = self._split_signal(signal, CHUNK_LEN, CHUNK_OVERLAPPING)

            signals = self._transformation(signals)

            # conver signals into batch of tensor
            signals = torch.cat(signals, 0)
            self._save_signals_features(signals, audio_path)
        else:
            signals = signals.to(self.device)

        with torch.no_grad():
            self.model.eval()
            embeddings = self.model(signals)            

        return embeddings.detach().numpy()


    def save_embeddings_data(self, embeddings: List[np.ndarray], save_path: str,
                replace_if_exist: bool = True) -> None:
        """Save embeddings to save_path in json line format
        Args:
            embeddings: list of array of embeddings
            save_path: path to save the embeddings
            replace_if_exist: if True, replace the existing file
            
        """
        if os.path.isfile(save_path) and replace_if_exist:
            os.remove(save_path)

        with open(save_path, 'w') as f:
            for line in embeddings:
                json.dump(line, f)
                f.write('\n')            

    def transform_data(self, ) -> None:
        """Embed all the original songs and hummed audios, then save to the 
        cached_dir, rewrite new file if already exist
        """

        # loop over all original songs, preprocess, forward it through model and
        # save all the embedding vectors to a file.

        for i, row in self.annotation.iterrows():
            print(f'Transforming audio {i+1}/{len(self.annotation)}',)
            song_embeddings = self._preprocess_and_embed_one_audio(
                            os.path.join(self.audio_dir, row['song_path']))

            hum_embeddings = self._preprocess_and_embed_one_audio(
                            os.path.join(self.audio_dir, row['hum_path']))

            song_data = {'id': row['music_id'], 'path': row['song_path'], 
                        'embeddings': song_embeddings.tolist()}
            
            hum_data = {'id': row['music_id'], 'path': row['hum_path'], 
                        'embeddings': hum_embeddings.tolist()}

            self.all_song_embeddings.append(song_data)
            self.all_hum_embeddings.append(hum_data)

        if self.save_embeddings:
            # if embedding file already exist, delete it
            song_embeddings_path = os.path.join(self.save_embeddings_path, 'val_original_song_embeddings.jl')
            hum_embeddings_path = os.path.join(self.save_embeddings_path, 'val_hummed_audio_embeddings.jl')

            self.save_embeddings_data(self.all_song_embeddings, song_embeddings_path)
            self.save_embeddings_data(self.all_hum_embeddings, hum_embeddings_path)

                


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    model = InceptionResnetV1(embedding_dims= EMBEDDING_DIMS, )
    evaluator = Evaluator(model, VAL_ANNOTATION_FILE, VAL_AUDIO_DIR, 
                        'l2', 'mel_spectrogram', SAMPLE_RATE,
                        SINGING_THRESHOLD, device, SAVE_EMBEDDING_PATH,
                        SAVE_FEATURES_PATH, True)

    evaluator.transform_data()