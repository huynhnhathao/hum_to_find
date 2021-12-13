import pickle
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

import arguments as args 

class CrepeDataset(Dataset):
    def __init__(self, 
                data_path: str,
                sample_len: int,
                scaler,
                device: str
                ):
        """Dataset class for CREPE features of audios

        example one sample: (song_tensor, hum_tensor, song_id)
        Args:
            annotation_path: path to annotation file
            data_path: path to all data. Expected to be like (filename, crepe_tensor)
            sample_len: number of frequencies for each sample.
                sample longer than this number will be cut, shorter
                will be padded.
            device: cpu or cuda 
        """
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
                self.data[i] = list(self.data[i])
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
                padding_size = self.sample_len - self.data[i][-2].shape[0]
                padding_ = np.zeros(padding_size)
                self.data[i][-2] = np.concatenate([self.data[i][-2], padding_]) 

            if self.data[i][-1].shape[0] < self.sample_len:
                padding_size = self.sample_len - self.data[i][-1].shape[0]
                padding_ = np.zeros(padding_size)
                self.data[i][-1] = np.concatenate([self.data[i][-1], padding_])                    


    def __getitem__(self, index):
        # random crop 4secs here
        index = index%len(self.data)
        item = self.data[index]
        # cut_point = np.random.randint(0, args.sample_len - args.chunk_len*100)
        song_freq = item[-2]
        hum_freq = item[-1]

        return (torch.tensor(song_freq, dtype=torch.float, device=self.device),
                torch.tensor(hum_freq, dtype=torch.float, device=self.device),
                torch.tensor(item[0], dtype=torch.long))

    def __len__(self):
        return len(self.data)*args.epoch_hack

if __name__ == '__main__':
    mydataset = CrepeDataset(args.train_data_path, args.sample_len, args.scaler, args.device)
    dataloader = torch.utils.data.DataLoader(mydataset, args.batch_size, shuffle = True)
    for song_tensor, hum_tensor, music_ids in dataloader:
        print(song_tensor.shape)
        print(hum_tensor.shape)
        print( music_ids)
        print(song_tensor)
        print(hum_tensor)
        print(len(mydataset))
        break
        