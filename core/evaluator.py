import logging
import json
import os
from typing import Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torchaudio

from hum_to_find.core.constants import *

logger = logging.getLogger()


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
                num_samples: int,
                singing_threshold: float,
                device: str,
                save_embeddings_path: str,
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
        """
        self.model = model
        self.annotation = pd.read_csv(annotation_file)
        self.audio_dir = audio_dir
        self.distance_method = distance_method
        self.transformation = self._get_transformation(transformation)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.singing_threshold = singing_threshold
        self.do_cache
        self.device = device
        self.save_embeddings_path = save_embeddings_path
        self.save_embeddings = save_embeddings

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

    def _split_signal(self, signal: torch.Tensor,
                    overlapping: int,
                    num_chunks: int) -> List[torch.Tensor]:

        """plit the signal into num_chunks equal chunks, 
        each chunk overlapping of overlapping samples
        Args:
            signal: the tensor signal to be splited
            overlapping: the number of samples overlapping between chunks
            num_chunks: number of chunks to be extracted

        """

        # right now the signal must has the size of (1, NUM_SAMPLE)
        assert signal.shape[1] == self.num_samples, 'unexpected num_samples'

        all_chunks = []
        for i in range(0, num_chunks):
            all_chunks.append(signal[:, i*overlapping: (i+2)*overlapping] )

        return all_chunks

    def _transformation(self, signals: List[torch.Tensor]) -> List[torch.Tensor]:
        """transform each signal in signals into mel-spectrogram"""
        transformed_signals = []
        for signal in signals:
            transformed_signals.append(self.transformation(signal))

        return transformed_signals



    def _preprocess_and_embed_one_audio(self,
                                        audio_path: str) -> np.ndarray:
        """Preprocess one audio, split it into chunks, transform it and forward
        it through the model
        Args:
            audio_path: relative path to the audio

        Returns: Embedding vectors for each chunk
        """

        signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_head_if_necessary(signal)
        signal = self._cut_tail_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self._split_signal(signal, 16000, NUM_CHUNKS_EACH_AUDIO)

        signals = self._transformation(signal)

        # conver signals into batch of tensor
        signals = torch.cat(signals, 0)
        
        with torch.no_grad():
            self.model.eval()
            embeddings = self.model(signals)            

        return embeddings.detach().numpy()


    def save_data(self, embeddings: List[np.ndarray], save_path: str,
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
        self.all_song_embeddings = []
        self.all_hum_embeddings = []
        for i, row in self.annotation.iterrows():
            
            song_embeddings = self._preprocess_and_transform_one_audio(row['song_path'])
            hum_embeddings = self._preprocess_and_transform_one_audio(row['hum_path'])

            song_data = {'id': row['song_id'], 'path': row['song_path'], 
                        'embeddings': song_embeddings}
            
            hum_data = {'id': row['song_id'], 'path': row['hum_path'], 
                        'embeddings': hum_embeddings}

            self.all_song_embeddings.append(song_data)
            self.all_hum_embeddings.append(hum_data)

        if self.save_embeddings:
            # if embedding file already exist, delete it
            song_embeddings_path = os.path.join(self.cached_path, 'original_song.jl')
            hum_embeddings_path = os.path.join(self.cached_path, 'hummed_audio.jl')

            self.save_data(self.all_song_embeddings, song_embeddings_path)
            self.save_data(self.all_hum_embeddings, hum_embeddings_path)

                


