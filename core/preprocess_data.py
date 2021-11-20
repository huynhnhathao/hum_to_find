import os

import torch
from torch.utils.data import Dataset
import pandas as pd
import torchaudio

# TODO: instead of cut the song sample on the right, let's cut it at random position
# inside the allowed range of samples, consider it as data augmentation


# TODO: cache transformed data into disk
# Test everything here, when run we bring it to colab to run on GPU

class HumDataset(Dataset):

    def __init__(self,
                 annotations_file: str,
                 audio_dir: str,
                 transformation,
                 target_sample_rate: int,
                 num_samples: int,
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

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, index: int):

        audio_sample_path = self._get_audio_sample_path(index)

        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label

    def _cut_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
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
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal: torch.Tensor) -> torch.Tensor:
        """mix down to one channel if there are more than one channel"""
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index: int) -> str:
        path = os.path.join(self.audio_dir, self.annotations.loc[
            index, 'path'])
        return path

    def _get_audio_sample_label(self, index: int) -> int:
        """return the id of the audio"""
        return self.annotations.loc[index, 'music_id']


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/huynhhao/Desktop/hum/hum_to_find/meta_data/train_annotation.csv"
    AUDIO_DIR = "/home/huynhhao/Desktop/hum/data"
    SECS = 10
    SAMPLE_RATE = 16000
    NUM_SAMPLES = 160000 

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device {device}")

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    usd = HumDataset(ANNOTATIONS_FILE,
                            AUDIO_DIR,
                            mel_spectrogram,
                            SAMPLE_RATE,
                            NUM_SAMPLES,
                            device)
    sampler = torch.utils.data.RandomSampler(usd)
    dataloader = torch.utils.data.DataLoader(usd, batch_size = 32, sampler = sampler)
    batch = next(iter(dataloader))
    print(len(batch))
    print(batch[0].shape)
    print(batch[1])