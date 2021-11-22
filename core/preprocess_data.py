import os
import logging
from typing import Tuple, List

import torch
from torch.utils.data import Dataset
import torchaudio

import pandas as pd
import numpy as np

from constants import *

# TODO: cache transformed data into disk
# Test everything here, when run we bring it to colab to run on GPU


logger = logging.getLogger()

class HumDataset(Dataset):

    def __init__(self,
                 annotations_file: str,
                 audio_dir: str,
                 transformation,
                 target_sample_rate: int,
                 num_samples: int,
                 singing_threshold: int,
                 device: str) -> None:

        """
        Args:
            annotations: path to the csv file contains label and path info
            audio_dir: path to the train dir
            transformation: transformation object
            target_sample_rate: the sample rate want to use
            num_sample: number of sample for each audio file
            device: cpu or cuda

        """
        
        self.annotations = pd.read_csv(annotations_file)
        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.singing_threshold = singing_threshold


        # My clever (or dirty?) way to solve the problem: only 2 secs of one audio
        # create one sample, so one (original, hum) tuple can create multiple
        # samples, but __getitem__ only return one (sample, label), how do we 
        # remember the old samples created but not yet used?
        # we save it into samples list, and retrieved it when samples still has 
        # data.
        self.samples = []
        
        self.preprocess_and_load_all_data()
        
        # keep track of the sample index in getitem
        self.sample_index = 0

    def __len__(self) -> int:
        # This method is just a dummy method, the random sampling job of the data
        # is left for this class, I dont know if there is a better way.
        return len(self.samples)

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

                this_label = label + i 

                original_sample = (original, this_label)
                hum_sample = (hum, this_label)

                self.samples.append(original_sample)
                self.samples.append(hum_sample)

        logger.info('Data loaded.')

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.samples[index]



    # def __getitem__(self, index: int
    #             ) -> Tuple[Tuple[torch.Tensor, torch.tensor], torch.Tensor]:

    #     """
    #     Each item indexing one song id, each song id has two audio: the
    #     original song and the hummed song. Each audio will go through:
    #         1. Pre-cut the left audio until it starts to sing
    #         2. Pad/cut each audio to has 10 second.
    #         3. Split the audio into 2-sec chunks, overlapping 1 sec.

    #     Finally, one song id/item will give 9 tuple of (original, hummed) sample.

    #     Returns:
    #         9 of this ((original, hummed), song_id)

    #     """
    #     if self.samples:
    #         return self.samples.pop()
    #     else:
    #         if self.random_indices:
    #             index = self.random_indices.pop()
    #             original_path, hum_path = self._get_audio_sample_path(index)
    #         else:
    #             self.random_indices = list(np.random.randint(0, len(self.annotations), len(self.annotations)))
    #             index = self.random_indices.pop()
    #             original_path, hum_path = self._get_audio_sample_path(index)

    #         label = self._get_audio_sample_label(index)

    #         original_signal, original_sr = torchaudio.load(original_path)
    #         original_signal = original_signal.to(self.device)
    #         original_signal = self._resample_if_necessary(original_signal, original_sr)
    #         original_signal = self._mix_down_if_necessary(original_signal)
    #         original_signal = self._cut_head_if_necessary(original_signal)
    #         original_signal = self._cut_tail_if_necessary(original_signal)
    #         original_signal = self._right_pad_if_necessary(original_signal)
    #         original_signals = self._split_signal(original_signal, 16000, NUM_CHUNKS_EACH_AUDIO)
        
    #         original_signals = self._transformation(original_signals)
            

    #         hum_signal, hum_sr = torchaudio.load(hum_path)
    #         hum_signal = hum_signal.to(self.device)
    #         hum_signal = self._resample_if_necessary(hum_signal, hum_sr)
    #         hum_signal = self._mix_down_if_necessary(hum_signal)
    #         hum_signal = self._cut_head_if_necessary(hum_signal)
    #         hum_signal = self._cut_tail_if_necessary(hum_signal)
    #         hum_signal = self._right_pad_if_necessary(hum_signal)
    #         hum_signals = self._split_signal(hum_signal, 16000, NUM_CHUNKS_EACH_AUDIO)

    #         hum_signals = self._transformation(hum_signals)

    #         for i, (original, hum) in enumerate( zip(original_signals, hum_signals)):
    #             # We don't want 2 sample of the same song to become (anchor, negative)
    #             # so we will do a trick into the label of sameples belong to the 
    #             # same song: every sameples belong the the same sone will has id
    #             # in range label + 9. with this trick, we can imply
    #             # which samples are actually belong to the same song.
    #             # because the id range are very large, so this should not be a problem
    #             # however, this is still a dirty workaround, until I know a better 
    #             # way. There will have cases where valid negative become invalid 
    #             # because of this trick, but it should not be a problem because
    #             # given the batch size and the random selection of sample.
    #             this_label = label + i 

    #             original_sample = (original, this_label)
    #             hum_sample = (hum, this_label)

    #             self.samples.append(original_sample)
    #             self.samples.append(hum_sample)
    #     return self.samples.pop()


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

    usd = HumDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            SINGING_THRESHOLD,
                            DEVICE)
    sampler = torch.utils.data.RandomSampler(usd)
    dataloader = torch.utils.data.DataLoader(usd, batch_size = 16, sampler = sampler)
    batch = next(iter(dataloader))
    print(len(batch))
    print(batch[0].shape)
    print(batch[1])