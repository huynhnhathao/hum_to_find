import torch
import torch.nn as nn
from torch import Tensor

import arguments as args
class LSTMEmbedder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                num_layers: int, dropout:bool,
                bidirectional: bool, proj_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                        num_layers=num_layers, dropout=dropout, 
                        bidirectional=bidirectional, proj_size=proj_size)

    def forward(self, x:Tensor) -> Tensor:
        out, (h_n, c_n) = self.lstm(x)
        return out



if __name__ == '__main__':
    mynet = LSTMEmbedder(args.input_size, args.hidden_size, args.num_layers,
                args.dropout, args.bidirectional, args.proj_size)