import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

import pandas as pd

class CrepeDataset(Dataset):
    def __init__(self, 
                data_path: str,
                sample_len: int,
                
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
        self.sample_len = sample_len
        # load all data to RAM
        self.data = torch.load(data_path)

    def __getitem__(self, index):
        return (torch.tensor(self.data[index][0], dtype=torch.float, device=self.device),
                torch.tensor(self.data[index][1], dtype=torch.float, device=self.device),
                torch.tensor(self.data[index][2], dtype=torch.long))

    def __len__(self):
        return len(self.data)