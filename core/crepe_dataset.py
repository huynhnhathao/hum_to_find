import pickle
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

class CrepeDataset(Dataset):
    def __init__(self, 
                data_path: str,
                sample_len: int,
                scaler,
                device: str
                ):
        """Dataset class for CREPE features of audios
        This dataset is expected to be padded and tupled with its label,
        no preprocess needed here.
        example one sample: (song_tensor, hum_tensor, song_id)
        Args:
            annotation_path: path to annotation file
            data_path: path to all data. Expected to be like (filename, crepe_tensor)
            sample_len: number of frequencies for each sample.
                sample longer than this number will be cut, shorter
                will be padded.
            device: cpu or cuda 
        """
        # self.annotation = pd.read_csv(annotation_path)
        self.data_path = data_path
        self.sample_len = sample_len
        self.scaler = scaler
        self.device = device
        # load all data to RAM
        self.data = pickle.load(open(data_path, 'rb'))  
        self._scale_data()
        self._cut_and_pad_if_necessary()
        
    def _scale_data(self,) -> None:
        """Scaling if self.scaler is not None, """
        for i in range(len(self.data)):
                self.data[i][-2] = self.scaler(self.data[i][-2])
                self.data[i][-1] = self.scaler(self.data[i][-1])

    def _cut_and_pad_if_necessary(self,)-> None:
        for i in range(len(self.data)):
            # cut tail if longer than self.sample_len
            if self.data[i][-2].shape[0] > self.sample_len:
                self.data[i][-2] = self.data[i][-2][:self.sample_len]
            if self.data[i][-1].shape[0] > self.sample_len:
                self.data[i][-1] = self.data[i][-1][:self.sample_len]
            # pad tail if shorter than self.sample_len
            if self.data[i][-2].shape[0] < self.sample_len:
                padding_size = self.sample_len - len(self.data[i][-2].shape[0])
                padding_ = np.zeros(padding_size)
                self.data[i][-2] = np.concatenate([self.data[i][-2], padding_]) 

            if self.data[i][-1].shape[0] < self.sample_len:
                padding_size = self.sample_len - len(self.data[i][-1].shape[0])
                padding_ = np.zeros(padding_size)
                self.data[i][-1] = np.concatenate([self.data[i][-1], padding_])                    


    def __getitem__(self, index):
        return (torch.tensor(self.data[index][-2], dtype=torch.float, device=self.device),
                torch.tensor(self.data[index][-1], dtype=torch.float, device=self.device),
                torch.tensor(self.data[index][0], dtype=torch.long))

    def __len__(self):
        return len(self.data)