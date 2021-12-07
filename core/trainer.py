import logging
from typing import Tuple, List, Union
import time
import os

import torch
import torchaudio
from torch import nn 
from torch.utils.data import DataLoader

import numpy as np
from core.constants import LOG_FILE_PATH

from triplet_mining_online import batch_hard_triplet_loss, batch_all_triplet_loss

LOG_FILE_PATH = r'C:\Users\ASUS\Desktop\repositories\hum_to_find\core'

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(LOG_FILE_PATH)
formmater = logging.Formatter('%(asctime)s - %(message)s')
stream_handler.setFormatter(formmater)
file_handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# TODO: evaluate on train datas
class Trainer:
    def __init__(self, model,
                loss_fn, optimizer, eval_each_num_epochs: int,
                checkpoint_epochs: int, epochs:int, device: str,
                save_model_path: str) -> None:
        """
        Trainer class to train the embedding model
        
        Args:
            model: pytorch model to train
            loss_fn:
            optimizer:
            eval_each_num_epochs: after this number of epochs, do one evaluation
                on val data
            checkpoint_epochs: save model after this number of epochs
            epochs: number of training epochs
            device: 
            save_model_path: path to save the model
        """


        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.eval_each_num_epochs = eval_each_num_epochs
        self.checkpoint_epochs = checkpoint_epochs
        self.epochs = epochs
        self.device = device
        self.save_model_path = save_model_path
        # save training time for each epoch to estimate remaining time
        self.epoch_time = []

    def train_single_epoch(self,) -> None:
        self.model.train()
        epoch_loss = []
        for music_ids, song_names, hum_names, song_tensor, hum_tensor in self.dataloader:
            # string target to int target
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            embeddings = self.model(inputs)
            loss, positive_rate = self.loss_fn(targets, embeddings, 1.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.detach().item())

        logger.info(f'loss: {sum(epoch_loss)/len(epoch_loss)}')

    def evaluate_on_train(self, ) -> None:
        """Evaluate model on val data, using mean reciprocal rank"""
        pass
    def evaluate_on_val(self, ) -> None:
        pass

    def save_model(self, current_epoch: Union[int, str]) -> None:
        """save the current model into self.save_model_path"""
        
        filename = f'inception_resnet_epoch{current_epoch}.pt'
        filepath = os.path.join(self.save_model_path, filename)
        logger.info(f'Saving model into {filepath}')
        torch.save(self.model.state_dict(), filepath)


    def train(self, )-> None:
        for i in range(self.epochs):
            start = time.time()
            logger.info(f"Epoch {i+1}")
            self.train_single_epoch()

            end = time.time()
            time_spent = (end - start)/60
            self.epoch_time.append(time_spent)
            logger.info(f"Estimated time per epoch: {np.mean(self.epoch_time)}-minutes")
            logger.info(f"Estimated remaining time: {(self.epochs - i - 1)*np.mean(self.epochs)}-minutes")
            if (i + 1) % self.eval_each_num_epochs == 0:
                self.evaluate() 
            if (i+1) % self.checkpoint_epochs == 0:
                self.save_model(i+1)
        
        self.save_model('last_epoch')
        logger.info('Finish training.')



if __name__ == '__main__':

    logger.info(f'Using {DEVICE}')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=TRANSFORMER_NFFT,
        hop_length=TRANSFORMER_HOP_LENGTH,
        n_mels=N_MELS,
        normalized = True
    )

    hds = HumDatasetNoSplit(TRAIN_ANNOTATIONS_FILE, TRAIN_AUDIO_DIR, mel_spectrogram, SAMPLE_RATE,
                    NUM_SAMPLES, SINGING_THRESHOLD, DEVICE, SAVE_TRAIN_FEATURES_PATH)

    evaluator = Evaluator(VAL_ANNOTATION_FILE, VAL_AUDIO_DIR, NUM_SAMPLES,
                    'euclidean', 'mel_spectrogram', SAMPLE_RATE,
                    SINGING_THRESHOLD, DEVICE, SAVE_EMBEDDING_PATH, 
                    SAVE_VAL_FEATURES_PATH, False, 1.1 )
    # random_sampler = torch.utils.data.RandomSampler(hds, )
    train_dataloader = DataLoader(hds, BATCH_SIZE, shuffle = False)

    inception_resnet = InceptionResnetV1(embedding_dims = EMBEDDING_DIMS ).to(DEVICE)

    loss_fn = batch_hard_triplet_loss
    optimizer = torch.optim.Adam(inception_resnet.parameters(), 
                                lr = LEARNING_RATE)
    trainer = Trainer(inception_resnet, evaluator, train_dataloader, loss_fn, optimizer,
                    EVAL_EACH_NUM_EPOCHS, CHECKPOINT_EPOCHS, EPOCHS, DEVICE,
                    SAVE_MODEL_PATH)
                    

    trainer.train()
