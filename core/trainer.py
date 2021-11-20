import logging
from typing import Tuple, List
import torch
import torchaudio
from torch import nn 
from torch.utils.data import DataLoader

import numpy as np 
from sklearn import preprocessing

from preprocess_data import HumDataset
from inception_resnet import *
from triplet_mining_online import batch_hard_triplet_loss, batch_all_triplet_loss
from constants import *

handler = logging.StreamHandler()
formmater = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class Trainer:
    
    def __init__(self, model, dataloader,
                loss_fn, optimizer, epochs:int, device: str) -> None:

        self.encoder = preprocessing.LabelEncoder()
        self.model = model
        self.dataloader = dataloader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = device

    def train_single_epoch(self,) -> None:
        epoch_loss = []
        _positive_fractions = []
        for inputs, targets in self.dataloader:
            # string target to int target
            targets = torch.tensor(self.encoder.fit_transform(targets))

            inputs, targets = inputs.to(device), targets.to(device)
            embeddings = self.model(inputs)
            loss, positive_rate = self.loss_fn(targets, embeddings, 1.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.detach().item())
            _positive_fractions.append(positive_rate)
            logger.info(f'positive triple fraction {(positive_rate)}')

        logger.info(f'loss: {sum(epoch_loss)/len(epoch_loss)}')
        logger.info(f'Epoch positive triple fraction {np.mean(_positive_fractions)}')


    def train(self, )-> None:
        for i in range(self.epochs):
            logger.info(f"Epoch {i+1}")
            self.train_single_epoch()
            
        logger.info('Finish training.')


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info(f'Using {device}')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=512,
        hop_length=256,
        n_mels=96
    )

    hds = HumDataset(TRAIN_ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,
                    NUM_SAMPLES, SINGING_THRESHOLD, DEVICE)


    random_sampler = torch.utils.data.RandomSampler(hds, )
    train_dataloader = DataLoader(hds, BATCH_SIZE, sampler = random_sampler)

    inception_resnet = InceptionResnetV1(embedding_dims = EMBEDDING_DIMS )

    loss_fn = batch_all_triplet_loss
    optimizer = torch.optim.Adam(inception_resnet.parameters(), 
                                lr = LEARNING_RATE)
    trainer = Trainer(inception_resnet, train_dataloader, loss_fn, optimizer,
                    EPOCHS, device)
    
    trainer.train()

    torch.save(trainer.model.state_dict('inception_resnet.pt'))

    logger.info('Finish the job.')