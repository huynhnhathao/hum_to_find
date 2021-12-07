import logging
from typing import Tuple, List, Union
import time
import os

import torch
from torch import nn 
from torch.utils.data import DataLoader
from crepe_dataset import CrepeDataset
from resnet1d import ResNet1D

import numpy as np

from triplet_mining_online import batch_hard_triplet_loss, batch_all_triplet_loss
import arguments as args
import faiss_comparer

stream_handler = logging.StreamHandler()
# file_handler = logging.FileHandler(args.log_file_path)
formmater = logging.Formatter('%(asctime)s - %(message)s')
stream_handler.setFormatter(formmater)
# file_handler.setFormatter(formmater)

logger = logging.getLogger()
logger.addHandler(stream_handler)
# logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# TODO: evaluate on train datas
class Trainer:
    def __init__(self, model,
                loss_fn, optimizer, dataloader, 
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
        self.dataloader = dataloader
        self.eval_each_num_epochs = eval_each_num_epochs
        self.checkpoint_epochs = checkpoint_epochs
        self.epochs = epochs
        self.device = device
        self.save_model_path = save_model_path
        # save training time for each epoch to estimate remaining time
        self.epoch_time = []

    def train_single_epoch(self, ) -> None:
        self.model.train()
        # collect embeddings to do evaluation on train
        song_embeddings = []
        hum_embeddings = []
        embedding_labels = []

        epoch_loss = []
        positive_rates = []
        for song_tensor, hum_tensor, music_ids in self.dataloader:
            # string target to int target
            # inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs = torch.cat((song_tensor, hum_tensor), dim=0, ).unsqueeze(1).to(self.device)
            targets = torch.cat((music_ids, music_ids), dim=0, ).to(self.device)
            embeddings = self.model(inputs)
        
            loss, positive_rate = self.loss_fn(targets, embeddings, 1.0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss.append(loss.detach().item())
            positive_rates.append(positive_rate)

            song_embeddings.append(embeddings[:args.batch_size, :].detach().cpu().numpy())
            hum_embeddings.append(embeddings[args.batch_size:, :].detach().cpu().numpy())
            embedding_labels.append(music_ids.detach().cpu().numpy())

        logger.info(f'loss: {sum(epoch_loss)/len(epoch_loss)}')
        logger.info(f'positive rate: {sum(positive_rates)/len(positive_rates)}')
        mrr = faiss_comparer.FaissEvaluator(args.embedding_dim, song_embeddings,
                        hum_embeddings, embedding_labels).evaluate()
        logger.info(f'train mrr: {mrr}')

    def evaluate_on_train(self, ) -> None:
        """Evaluate model on val data, using mean reciprocal rank"""
        pass
    def evaluate_on_val(self, ) -> None:
        pass

    def save_model(self, current_epoch: Union[int, str]) -> None:
        """save the current model into self.save_model_path"""
        
        filename = f'model_epoch{current_epoch}.pt'
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
            logger.info(f"Estimated remaining time: {(self.epochs - i - 1)*np.mean(self.epoch_time)}-minutes")

            # if (i + 1) % self.eval_each_num_epochs == 0:
            #     self.evaluate_on_train() 
            if (i+1) % self.checkpoint_epochs == 0:
                self.save_model(i+1)
        
        self.save_model('last_epoch')
        logger.info('Finish training.')



if __name__ == '__main__':

    logger.info(f'Using {args.device}')



    mydataset = CrepeDataset(args.train_data_path, args.sample_len, args.scaler, 
                    args.device)
    train_dataloader = DataLoader(mydataset, args.batch_size, shuffle = True)
    model = ResNet1D(1, args.base_filters, args.kernel_size, args.stride,
                args.groups, args.n_blocks, args.embedding_dim, )

    loss_fn = batch_all_triplet_loss
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr = args.learning_rate)
    trainer = Trainer(model, loss_fn, optimizer, train_dataloader,
                    args.eval_each_num_epochs, args.checkpoint_epochs, args.epochs,
                    args.device, args.save_model_path).to(args.device)
                    
    trainer.train()
