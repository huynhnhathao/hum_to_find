import logging

import torch
import torchaudio
from torch import nn 
from torch.utils.data import DataLoader

import numpy as np 

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


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimizer, device) -> None:
    epoch_loss = []
    _positive_fractions = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        embeddings = model(inputs)
        loss, positive_rate = loss_fn(targets, embeddings, 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.detach().item())
        _positive_fractions.append(positive_rate)
        logger.info(f'Epoch positive triple fraction {(positive_rate)}')

    logger.info(f'loss: {sum(epoch_loss)/len(epoch_loss)}')
    logger.info(f'Epoch positive triple fraction {np.mean(_positive_fractions)}')


def train(model, data_loader, loss_fn, optimizer, device, epochs)-> None:
    for i in range(epochs):
        logger.info(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        
    logger.info('Finish training.')


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info(f'Using {device}')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=96
    )

    hds = HumDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,
                    NUM_SAMPLES, device)


    random_sampler = torch.utils.data.RandomSampler(hds, )
    train_dataloader = DataLoader(hds, BATCH_SIZE, sampler = random_sampler)

    inception_resnet = InceptionResnetV1(embedding_dims = EMBEDDING_DIMS )

    loss_fn = batch_all_triplet_loss
    optimizer = torch.optim.Adam(inception_resnet.parameters(), 
                                lr = LEARNING_RATE)

    train(inception_resnet, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(inception_resnet.state_dict('inception_resnet.pt'))

    logger.info('Finish the job.')