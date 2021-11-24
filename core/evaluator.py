import logging
import json
import os
import collections
from re import L
from typing import Any, List, Tuple, Union, Dict

import numpy as np
import pandas as pd

import torch
import torchaudio
from sklearn import neighbors

from constants import *
from inception_resnet import *

logger = logging.getLogger()
# if not logger.hasHandlers():
#     handler = logging.StreamHandler()
#     formmater = logging.Formatter('%(asctime)s - %(message)s')
#     handler.setFormatter(formmater)
#     logger.addHandler(handler)
#     logger.setLevel(logging.INFO)

# TODO: Preprocess and save all data features into one file.
# NOTE: get song id will be broken if change val_annotation file, need manual ajusted




class Evaluator:
    """
    Evaluator class evaluate the given model on the given dataset, using `l2` distance
    or `adaptive`
    """
    def __init__(self,
                annotation_file: str,
                audio_dir: str,
                num_samples: int, 
                distance_method: str,
                transformation: str,
                target_sample_rate: int, 
                singing_threshold: float,
                device: str,
                save_embeddings_path: str,
                save_val_features_path: str,
                save_embeddings: bool = False,
                matched_threshold: float = 1.1
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
            match_threshold: the threshold to consider two sample are match, when
                compare in Euclidean distance
        """
        self.annotation = pd.read_csv(annotation_file)
        self.annotation_path = annotation_file
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        self.distance_method = distance_method
        
        self.target_sample_rate = target_sample_rate
        self.singing_threshold = singing_threshold
        self.device = device
        self.save_embeddings_path = save_embeddings_path
        self.save_val_features_path = save_val_features_path
        self.save_embeddings = save_embeddings
        self.matched_threshold = matched_threshold
        self.transformation = self._get_transformation(transformation)
        # save all song and hum embeddings to measure distances later

        self.val_samples = {}
        self.val_embeddings = {}

    def _get_transformation(self, transformation: str) -> Any:
        transformer = None
        if transformation == 'mel_spectrogram':
            transformer = torchaudio.transforms.MelSpectrogram(
                            sample_rate = SAMPLE_RATE,
                            n_fft = TRANSFORMER_NFFT,
                            hop_length = TRANSFORMER_HOP_LENGTH,
                            n_mels = N_MELS, normalized=True)
        if transformer is None:
            raise ValueError('Transformer not specified')
        return transformer.to(self.device)

    def _resample_if_necessary(self, signal: torch.Tensor, sr: int) -> torch.Tensor:
        """Resample the signal with sample rate = sr into self.target_sample_rate"""
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(self.device)
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

        torch.save(signals.to('cpu'), os.path.join(self.save_features_path, filename))
        
    def _preprocess_one_audio(self, audio_path) -> torch.Tensor:
                                        

        signal, sr = torchaudio.load(audio_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        # signal = self._cut_head_if_necessary(signal) No cut head when evaluating
        signal = self._cut_tail_if_necessary(signal) 
        signal = self._right_pad_if_necessary(signal) 

        signal = self.transformation(signal).unsqueeze(0)    

        return signal

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

    def transform_val_data(self, ) -> None:
        """ Transform all val data and save to disk, or reload from disk if exist
        """

        if os.path.isfile(self.save_val_features_path):
            logger.info(f'Loading all val samples from {self.save_val_features_path}')
            self.all_val_features = torch.load(self.save_val_features_path)
        else:
            for _, row in self.annotation.iterrows():
                
                song_features = self._preprocess_one_audio(
                                os.path.join(self.audio_dir, row['song_path']))

                hum_features = self._preprocess_one_audio(
                                os.path.join(self.audio_dir, row['hum_path']))

                key = row['music_id']

                self.val_samples[key] = ((song_features, row['song_path']), (hum_features, row['hum_path']))

            logger.info(f'Saving self.val_samples into {self.save_val_features_path}')

            torch.save(self.val_samples, self.save_val_features_path)

    def _compute_embeddings(self, model) -> None:
        """Compute all embeddings for self.val_samples"""
        
        self.val_embeddings = {}
        model.eval()
        with torch.no_grad():
            for label, item in self.val_samples.items():
                song_features = item[0][0]
                hum_features = item[1][0]
                song_embedding = model(song_features).detach().cpu().numpy()
                hum_embedding = model(hum_features).detach().cpu().numpy()
                self.val_embeddings[label] = ((song_embedding, item[0][1]), (hum_embedding, item[1][1]))

    def query_one_hum(self, hum_embeddings: Dict[str, Any],
                        knn: neighbors.KNeighborsClassifier,
                        ) -> List[Union[str, int]]:
        """
        Compare the hum_embeddings to the database embeddings and return 10
        most likely song id.

        For each chunk's embedding in the hum, we can query 10 nearest neighbors
        of that chunk's embedding, which means if we have 10 chunk we will have 
        100 neighbors. Then we will rank those neighbors base on the number of times
        they appear in our 100 neigbors, and select top 10 neighbors.

        # NOTE: distance compare are ideas, not yet implemented.
        If neighbors has the same rank, we will use their distance to rank them, 
        The smaller the distance, the higher the rank of that neigbor.

        Returns:  The return list is like: [0000.mp3, 0, 1, 2, ..., 9], where the
            elements from [1:] are the indices of the songs in our database, note 
            that they are the indices, not the song ids
        
        """
        hum_filename = hum_embeddings[1].split('/')[-1]
        result = [hum_filename, ]
        embedding = hum_embeddings[0]

        embedding = np.array(embedding).reshape(-1, EMBEDDING_DIMS)
        distances, neighbors = knn.kneighbors(embedding, NUM_SONG_RETRIEVED_PER_QUERY)


        song_ids = [self.database_df.loc[index, 'id'] for index in neighbors]

        result.extend(song_ids)
        return result


    def knn_retriever(self, ) -> List[List[Any]]:
        """Use KNN to find the K nearest neighbors to the query embeddings
        """
        # construct a dataframe of song_id and embeddings to train knn classifier
        
        self.database_df = pd.DataFrame([], columns = ['id', 'embedding'])
        for key, item in self.val_embeddings.items():

            row = {'id': key, 'embedding': item[0][0]}
            self.database_df = self.database_df.append(row, ignore_index=True)
        
        knn = neighbors.KNeighborsClassifier(n_neighbors= 10, weights= 'distance', 
                                    metric= 'euclidean')

        df_data = np.vstack(self.database_df['embedding'].values)
        labels = self.database_df['id'].astype(int).values
        knn.fit(df_data, labels)


        # preds is a list of list contains all predictions for every hum audio 
        # in the val set. The inner list: [hum_file_name, pred1, pred2,..,pred10]
    
        predictions = []
        for key, item in self.val_embeddings:
            pred = self.query_one_hum(item[1], knn )
            predictions.append(pred)
        return predictions

    def get_song_id(self, hum_file_name: str) -> int:
        """Get the song id of a hummed audio from the annotation file"""
        hum_path = 'train/hum/'+hum_file_name
        song_id = self.annotation.loc[self.annotation['hum_path'] == hum_path, 'music_id'].values[0]
        return song_id

    def compute_mean_reciprocal_rank(self,
                        predictions: List[List[Any]]) -> float:
        all_rr = []
        for pred in predictions:
            true_song_id = self.get_song_id(pred[0])
            if true_song_id in pred:
                position = pred.index(true_song_id)
                rank = 1/position
                all_rr.append(rank)
            else:
                all_rr.append(0)
        return np.mean(all_rr)

    def evaluate(self, retrieving_method: str = 'knn',
                create_df: bool = False ) -> pd.DataFrame:
        """
        This method transform, embed and compare embeddings to retrieve song 
        for each hummed audios.

        Args:
        retrieving_method: method use to compare embeddings, right now only knn
            is available.
        create_df: if True, create and return a dataframe using the predictions data
        """

        if not self.val_samples:
            self.transform_val_data()

        logger.info('Computing embedding for val data')
        self._compute_embeddings()

        logger.info(f'Building and retrieving songs using {retrieving_method}')

        predictions = self.knn_retriever()

        logger.info('Computing mean reciprocal rank')
        mrr = self.compute_mean_reciprocal_rank( predictions)
        logger.info(f'Mean reciprocal rank = {mrr}')

        return None


        
if __name__ == '__main__':

    model = InceptionResnetV1(embedding_dims= EMBEDDING_DIMS, )
    evaluator = Evaluator(VAL_ANNOTATION_FILE, VAL_AUDIO_DIR, NUM_SAMPLES,
                        'euclidean', 'mel_spectrogram', SAMPLE_RATE,
                        SINGING_THRESHOLD, DEVICE, SAVE_EMBEDDING_PATH,
                        SAVE_VAL_FEATURES_PATH, False, MATCHED_THRESHOLD)

    evaluator.evaluate()