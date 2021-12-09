import logging
import pickle
from typing import Tuple, List, Union, Any
import time
import os

import torch
from torch import Tensor
from torch import nn 
from torch.utils.data import DataLoader
from crepe_dataset import CrepeDataset
from resnet1d import ResNet1D

import numpy as np

from triplet_mining_online import batch_hard_triplet_loss, batch_all_triplet_loss
import arguments as args
import faiss_comparer

stream_handler = logging.StreamHandler()
file_handler = logging.FileHandler(args.log_file_path)
formmater = logging.Formatter('%(asctime)s - %(message)s')
stream_handler.setFormatter(formmater)
file_handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# TODO: evaluate on train datas method
class Trainer:
    def __init__(self, model,
                loss_fn, optimizer, train_dataloader, val_data_path: str,
                train_data_path: str,
                eval_each_num_epochs: int,
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
        self.train_dataloader = train_dataloader
        self.val_data_path = val_data_path
        self.train_data_path = train_data_path

        self.eval_each_num_epochs = eval_each_num_epochs
        self.checkpoint_epochs = checkpoint_epochs
        self.epochs = epochs
        self.device = device
        self.save_model_path = save_model_path
        # save training time for each epoch to estimate remaining time
        self.epoch_time = []

        self.val_data = None
        self.train_data = None

        if self.val_data_path is not None:
            self.val_data = self._preprocess_val_data(self.val_data_path)
        
        if self.train_data_path is not None:
            self.train_data = self._preprocess_val_data(self.train_data_path)


    def _preprocess_val_data(self,
            data_path: str) -> List[Tuple[List[int], List[int], Tensor, Tensor]]:
        """Load val data"""
        val_data = pickle.load(open(data_path, 'rb'))
        
        song_labels = []
        hum_labels = []

        song_tensors = []
        hum_tensors = []

        for sample in val_data:

            song_freq = args.scaler(sample[-2])
            hum_freq = args.scaler(sample[-1])

            # split song freq
            while len(song_freq) > args.chunk_len*100:
                song_tensors.append(song_freq[:args.chunk_len*100])
                song_freq = song_freq[args.chunk_len*100:]
                song_labels.append(sample[0])

            # pad the last chunk
            padding_len = args.chunk_len*100 - len(song_freq)
            pad_ = np.zeros(padding_len)
            song_tensors.append(np.concatenate([song_freq, pad_]))
            song_labels.append(sample[0])
            # split hum freq

            while len(hum_freq) > args.chunk_len*100:
                hum_tensors.append(hum_freq[:args.chunk_len*100])
                hum_freq = hum_freq[args.chunk_len*100:]
                hum_labels.append(sample[0])

            padding_len = args.chunk_len*100 - len(hum_freq)
            pad_ = np.zeors(padding_len)
            hum_tensors.append(np.concatenate([hum_freq, pad_]))
            hum_labels.append(sample[0])

        song_tensors = [torch.tensor(x, dtype=torch.float32, device = self.device).unsqueeze(0).unsqueeze(0) for x in song_tensors]
        hum_tensors = [torch.tensor(x, dtype = torch.float32, device = self.device).unsqueeze(0).unsqueeze(0) for x in hum_tensors]

        song_tensors_ = torch.cat(song_tensors, dim=0)
        hum_tensors_ = torch.cat(hum_tensors, dim = 0)
        # song_tensors_ and hum_tensors_ now are batchs of embeddings with shape (batch, 1, features_dim)
        return (song_labels, hum_labels, song_tensors_, hum_tensors_)

    def train_single_epoch(self, eval_on_train: bool ) -> None:
        self.model.train()
        # collect embeddings to do evaluation on train
        epoch_loss = []
        positive_rates = []
        for song_tensor, hum_tensor, music_ids in self.train_dataloader:
            # string target to int target
            # inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = torch.cat((song_tensor, hum_tensor), dim=0, ).unsqueeze(1).to(self.device)
            targets = torch.cat((music_ids, music_ids), dim=0, ).to(self.device)
            embeddings = self.model(inputs)
        
            loss, positive_rate = self.loss_fn(targets, embeddings, 2.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.detach().item())
            positive_rates.append(positive_rate)

        logger.info(f'loss: {sum(epoch_loss)/len(epoch_loss)}')
        logger.info(f'positive rate: {sum(positive_rates)/len(positive_rates)}')
        if eval_on_train:
            self.evaluate_on_train()

    def evaluate_on_train(self, ) -> None:
        """Evaluate model on val data, using mean reciprocal rank"""
        self.model.eval()
        with torch.no_grad():
            song_embeddings = self.model(self.train_data[-2].detach().cpu().numpy())
            hum_embeddings = self.model(self.train_data[-1].detach().cpu().numpy())

        mrr = faiss_comparer.FaissEvaluator(args.embedding_dim, song_embeddings, 
                hum_embeddings, self.train_data[0], self.train_data[1])
        logger.info(f'MRR ON TRAIN" {mrr}')
    
    def evaluate_on_val(self, ) -> None:
        self.model.eval()
        with torch.no_grad():
                song_embeddings = self.model(self.val_data[-2]).detach().cpu().numpy()
                hum_embeddings = self.model(self.val_data[-1]).detach().cpu().numpy()     

        mrr = faiss_comparer.FaissEvaluator(args.embedding_dim, song_embeddings,
                        hum_embeddings, self.val_data[0], self.val_data[1]).evaluate()

        logger.info(f'MRR ON VAL: {mrr}')

    def save_model(self, current_epoch: Union[int, str]) -> None:
        """save the current model into self.save_model_path"""
        
        filename = f'model_epoch{current_epoch}.pt'
        filepath = os.path.join(self.save_model_path, filename)
        logger.info(f'Saving model into {filepath}')
        torch.save(self.model.state_dict(), filepath)


    def train(self, )-> None:
        for i in range(self.epochs):
            # start = time.time()
            logger.info(f"Epoch {i+1}/{args.epochs}")
            
            self.train_single_epoch(True)
            # end = time.time()
            # time_spent = (end - start)/60
            # self.epoch_time.append(time_spent)
            # logger.info(f"Estimated time per epoch: {np.mean(self.epoch_time)}-minutes")
            # logger.info(f"Estimated remaining time: {(self.epochs - i - 1)*np.mean(self.epoch_time)}-minutes")

            # if (i + 1) % self.eval_each_num_epochs == 0:
            #     self.evaluate_on_train() 
            if (i+1) % self.checkpoint_epochs == 0:
                self.save_model(i+1)
            if (i+1) % args.eval_each_num_epochs == 0:
                self.evaluate_on_val()
        
        self.save_model('last_epoch')
        logger.info('Finish training.')

if __name__ == '__main__':

    logger.info(f'Using {args.device}')

    mydataset = CrepeDataset(args.train_data_path, args.sample_len, args.scaler, 
                    args.device)

    train_dataloader = DataLoader(mydataset, args.batch_size, shuffle = True)

    model = ResNet1D(1, args.base_filters, args.kernel_size, args.stride,
                args.groups, args.n_blocks, args.embedding_dim, ).to(args.device)

    if args.pretrained_model_path is not None:
        logger.info(f'Load model state_dict from {args.pretrained_model_path}')
        model.load_state_dict(torch.load(args.pretrained_model_path))

    loss_fn = batch_all_triplet_loss
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr = args.learning_rate)
    trainer = Trainer(model, loss_fn, optimizer, train_dataloader, args.val_data_path,
                    args.train_data_path, args.eval_each_num_epochs,
                    args.checkpoint_epochs, args.epochs,
                    args.device, args.save_model_path)
                    
    trainer.train()
