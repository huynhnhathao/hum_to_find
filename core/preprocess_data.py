import os
import logging
import copy
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import numpy as np

from constants import *

# TODO: let the random chosing batch sample to this class, not the sampler
# then,for each batch we force it to not have more than 3 positive samples.

logger = logging.getLogger()

class HumDataset(Dataset):

    def __init__(self,
                 batch_size: int,
                 annotations_file: str,
                 audio_dir: str,
                 transformation,
                 target_sample_rate: int,
                 num_samples: int,
                 singing_threshold: int,
                 device: str,
                 save_features_path: str) -> None:

        """
        Args:
            batch_size: because this class handles the sampling process, so it 
                must know the batch size
            annotations: path to the csv file contains label and path info
            audio_dir: path to the train dir
            transformation: transformation object
            target_sample_rate: the sample rate want to use
            num_sample: number of sample for each audio file
            device: cpu or cuda

        Random sampling samples from this dataset is handled by this dataset,
        not by the torch.utils.data.Sampler object. This class will itself define
        its way to return batch of samples, such that:
            1. In one batch, one label has no more than two positive samples.
            2. one epoch will traverse all the dataset.

        """
        self.batch_size = batch_size
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.singing_threshold = singing_threshold
        self.save_features_path = save_features_path

        self.samples = {}
        if not self._load_cached_if_exist():
            self.preprocess_and_load_all_data()

        # plan and save the indices of sample in one epoch
        self.all_labels = list(self.samples.keys())
        self.next_sample = None
        self._plan_one_batch()

    def __len__(self) -> int:
        # This method is just a dummy method, the random sampling job of the data
        # is left for this class, I dont know if there is a better way.
        return len(self.samples)*2
    
    def _plan_one_batch(self) -> None:
        """
        Data return by this class will follow that epoch_indices, where each batch
        of samples must sastisfy some rules.
            1. In one batch, one label has no more than two positive samples.
            2. one epoch will traverse all the dataset.

        batch size should divided by two
        last batch may has less than batch_size samples
        """
        # logger.info('Planing one batch...')
        batch_labels = []

        while len(batch_labels) < self.batch_size//2:
            index = np.random.randint(0, len(self.all_labels))
            all_first_label = [x.split('_')[0] for x in batch_labels]
            first_label = self.all_labels[index].split('_')[0]
            if first_label in all_first_label:
                continue
            else:
                batch_labels.append(self.all_labels[index])            

        self.batch_labels = batch_labels

    def _load_cached_if_exist(self,)-> bool:
        """Load data in self.save_features_path if exist"""

        if os.path.isfile(self.save_features_path):
            logger.info(f'Loading train features data from {self.save_features_path}')
            self.samples = torch.load(self.save_features_path)
            return True

        return False

    def preprocess_and_load_all_data(self) -> None:
        """This method preprocess and load all data into memory, save to self.samples"""

        logger.info('Loading all data to memory...')
        for index in range(len(self.annotations)):

            original_path, hum_path = self._get_audio_sample_path(index)
            label = self._get_audio_sample_label(index)

            original_signal, original_sr = torchaudio.load(original_path)
            original_signal = original_signal.to(self.device)
            original_signal = self._resample_if_necessary(original_signal, original_sr)
            original_signal = self._mix_down_if_necessary(original_signal)
            original_signal = self._cut_head_if_necessary(original_signal)
            original_signal = self._cut_tail_if_necessary(original_signal)
            original_signal = self._right_pad_if_necessary(original_signal)
            original_signals = self._split_signal(original_signal, 16000, NUM_CHUNKS_EACH_AUDIO)
        
            original_signals = self._transformation(original_signals)

            hum_signal, hum_sr = torchaudio.load(hum_path)
            hum_signal = hum_signal.to(self.device)
            hum_signal = self._resample_if_necessary(hum_signal, hum_sr)
            hum_signal = self._mix_down_if_necessary(hum_signal)
            hum_signal = self._cut_head_if_necessary(hum_signal)
            hum_signal = self._cut_tail_if_necessary(hum_signal)
            hum_signal = self._right_pad_if_necessary(hum_signal)
            hum_signals = self._split_signal(hum_signal, 16000, NUM_CHUNKS_EACH_AUDIO)

            hum_signals = self._transformation(hum_signals)

            for i, (original, hum) in enumerate( zip(original_signals, hum_signals)):

                this_label = str(label)

                original_sample = (original, this_label)
                hum_sample = (hum, this_label)

                key = this_label + f'_{i}'
                if key in self.samples.keys(): # key conflict, create new
                    j = 9
                    new_key = this_label + f'_{i+j}'
                    while new_key in self.samples.keys():
                        j+= 1
                        new_key = this_label + f'_{i+j}'
                        if j > 1000:
                            logger.info(f'j = {j} too large')

                    key = new_key
                self.samples[key] = (original_sample, hum_sample)


        self.save_features_data()
        logger.info('Data loaded.')

    def save_features_data(self,) -> None:
        """Save all transformed features of samples to self.save_features_path"""
        logger.info(f'Saving all features data into {self.save_features_path}')
        torch.save(self.samples, self.save_features_path)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Return the next_sample if next_sample is not None, 
        else it pop one label in self.all_labels, return the first sample of that 
        label, save the remaining sample to self.last_sample
        """
        return_sample = None
        if self.next_sample is not None:
            return_sample = self.next_sample
            self.next_sample = None

        else:
            if self.batch_labels:
                label = self.batch_labels.pop(0)
                return_sample, self.next_sample = self.samples[label]
            else:
                self._plan_one_batch()
                label = self.batch_labels.pop(0)
                return_sample, self.next_sample = self.samples[label]

        return return_sample[0], int(return_sample[1])


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

    def _get_audio_sample_path(self, index: int) -> Tuple[str, str]:
        """Return the indices of the original and hummed audio"""

        original = os.path.join(self.audio_dir, self.annotations.loc[index, 'song_path'])
        hum = os.path.join(self.audio_dir, self.annotations.loc[index, 'hum_path'])
        return original, hum

    def _get_audio_sample_label(self, index: int) -> int:
        """return the id of the audio"""
        return self.annotations.loc[index, 'music_id']

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

if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/huynhhao/Desktop/hum/hum_to_find/meta_data/train_annotation.csv"
    AUDIO_DIR = "/home/huynhhao/Desktop/hum/data"
    SECS = 10
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 160000 


    print(f"Using device {DEVICE}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=TRANSFORMER_NFFT,
        hop_length=TRANSFORMER_HOP_LENGTH,
        n_mels=N_MELS
    )

    hds = HumDataset(TRAIN_ANNOTATIONS_FILE, TRAIN_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,
                    NUM_SAMPLES, SINGING_THRESHOLD, DEVICE, SAVE_TRAIN_FEATURES_PATH)
    # sampler = torch.utils.data.RandomSampler(usd)
    dataloader = torch.utils.data.DataLoader(hds, batch_size = 16,)
    for inputs, labels in dataloader:
        print(inputs)
        print(labels)