import logging

import torch
import torchaudio
from torch import nn 
from torch.utils.data import DataLoader
from core.preprocess_data import ANNOTATIONS_FILE, AUDIO_DIR

from preprocess_data import HumDataset
from inception_resnet import *
from triplet_mining_online import batch_hard_triplet_loss, batch_all_triplet_loss

handler = logging.StreamHandler()
formmater = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
EMBEDDING_DIMS = 512


ANNOTATIONS_FILE = r'C:\Users\ASUS\Desktop\hum\data\train\train_annotation.csv'
SAMPLE_RATE = 16000
NUM_SAMPLES = 160000


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimizer, device) -> None:
    epoch_loss = []
    for inputs, targets in data_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        embeddings = model(inputs)
        loss = loss_fn(targets, embeddings, 1.0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.detach().item())

    logger.info(f'loss: {sum(epoch_loss)/len(epoch_loss)}')


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
        n_mels=64
    )

    hds = HumDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,
                    NUM_SAMPLES)


    random_sampler = torch.utils.data.RandomSampler(hds, )
    train_dataloader = DataLoader(hds, BATCH_SIZE, sampler = random_sampler)

    inception_resnet = InceptionResnetV1(embedding_dims = EMBEDDING_DIMS )

    loss_fn = batch_all_triplet_loss
    optimizer = torch.optim.Adam(inception_resnet.parameters(), 
                                lr = LEARNING_RATE)

    train(inception_resnet, train_dataloader, loss_fn, optimizer, device, EPOCHS)

    torch.save(inception_resnet.state_dict('inception_resnet.pt'))

    logger.info('Finish the job.')