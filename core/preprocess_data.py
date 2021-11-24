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


logger = logging.getLogger()

class HumDatasetNoSplit(Dataset):

    def __init__(self,
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

        This class implement the sample with replacement interface of the dataset.

        """
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

        self.all_keys = list(self.samples.keys())

        self.next_sample = None

    def __len__(self) -> int:
        # This method is just a dummy method, the random sampling job of the data
        # is left for this class, I dont know if there is a better way.
        return len(self.samples)*4
    

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
            # original_signal = self._cut_head_if_necessary(original_signal)
            original_signal = self._cut_tail_if_necessary(original_signal)
            original_signal = self._right_pad_if_necessary(original_signal)
        
            original_signal = self._transformation(original_signal)

            hum_signal, hum_sr = torchaudio.load(hum_path)
            hum_signal = hum_signal.to(self.device)
            hum_signal = self._resample_if_necessary(hum_signal, hum_sr)
            hum_signal = self._mix_down_if_necessary(hum_signal)
            # hum_signal = self._cut_head_if_necessary(hum_signal)
            hum_signal = self._cut_tail_if_necessary(hum_signal)
            hum_signal = self._right_pad_if_necessary(hum_signal)

            hum_signal = self._transformation(hum_signal)
            
            if label not in self.samples.keys():
                self.samples[str(label)] = ((original_signal, label), (hum_signal, label))
            else:
                j = 1
                new_label = str(label) + f'_{j}'
                while  new_label in self.samples.keys():
                    j+=1
                    new_label = str(label) + f'_{j}'

                self.samples[new_label] = ((original_signal, label), (hum_signal, label))


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
            random_index = np.random.randint(0, len(self.samples))
  
            return_sample, self.next_sample = self.samples[self.all_keys[random_index]]

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

    def _transformation(self, signal: List[torch.Tensor]) -> List[torch.Tensor]:
        """transform each signal in signals into mel-spectrogram"""
        return self.transformation(signal)




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
        n_mels=N_MELS,
        normalized=True
    )

    hds = HumDatasetNoSplit(TRAIN_ANNOTATIONS_FILE, TRAIN_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,
                    NUM_SAMPLES, SINGING_THRESHOLD, DEVICE, SAVE_TRAIN_FEATURES_PATH)
    # sampler = torch.utils.data.RandomSampler(usd)
    dataloader = torch.utils.data.DataLoader(hds, batch_size = 16,)
    for inputs, labels in dataloader:
        print(inputs)
        print(labels)