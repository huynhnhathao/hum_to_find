from typing import List, Tuple, Dict
import pickle

import torch
import faiss
import pandas as pd
import numpy as np
from torch import Tensor
from hum_to_find.core import arguments as args

class Inferencer:
    def __init__(self, model_path: str, song_freq_path:str,
                hum_freq_path: str, sample_len: int, embedding_dim: int = 512 ) -> None:
        """
        Inferencer class do inference using the given path
        """
        self.model = self._load_model(model_path)
        self.song_freq = self._load_data(song_freq_path)
        self.hum_freq = self._load_data(hum_freq_path)

        self.sample_len = sample_len
        self.embedding_dim = embedding_dim

    def _load_model(self, path_to_model: str)->torch.nn.Module:
        model = ResNet1D(1, args.base_filters, args.kernel_size, args.stride,
                    args.groups, args.n_blocks, args.embedding_dim, )
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        return model

    def _load_data(self, path_to_data: str) -> List[Tuple[str, np.ndarray]]:
        data = pickle.load(open(path_to_data, 'rb'))
        return data

    def _preprocess_hum_freq(self, ) -> Dict[str, np.ndarray]:
        """Cut and pad all hum freq if necessary
        """
        hum_data = {}
        for path, freq in self.hum_freq:
            filename = path.split('/')[-1]
            freq = args.scaler(freq)

            if len(freq) > self.sample_len:
                freq = freq[:self.sample_len]
                hum_data[filename] = freq
            elif len(freq) < self.sample_len:
                pad_len = self.sample_len - len(freq)
                pad_ = np.zeros(pad_len)
                freq = np.concatenate([freq, pad_], axis = 0)
                hum_data[filename] = freq

        return hum_data

    def _preprocess_song_freq(self,
            hop_len: int = 100)-> Tuple[List[int], List[np.ndarray]]:
        """For each song freq, cut it to multiple song freqs with 
        len = sample_len. 
        Returns two list, one list of labels and one list of freqs such that 
            the song freq has label in labels with the same index
        """
        # we have to normalize the count by the total number of fragments for each
        # song
        song_ids = []
        song_freq = []

        for path, freq in self.song_freq:
            freq = args.scaler(freq)
            music_id = path.split('/')[-1].split('.')[0]
            if len(freq) <= self.sample_len:
                pad_len = self.sample_len - len(freq)
                pad_ = np.zeros(pad_len)
                freq = np.concatenate([freq, pad_], axis = 0)
                song_ids.append(music_id)
                song_freq.append(freq)
                
            else:
                while len(freq) > self.sample_len:
                        fragment = freq[:self.sample_len]
                        song_ids.append(music_id)
                        song_freq.append(fragment)
                        freq = freq[hop_len:]

                # Final fragment        
                pad_len = self.sample_len - len(freq)
                pad_ = np.zeros(pad_len)
                freq = np.concatenate([freq, pad_], axis = 0)
                song_ids.append(music_id)
                song_freq.append(freq)
        return song_ids, song_freq

    def _get_embeddings(self, data: Tensor) -> np.ndarray:
        """run inference of batch of data and return embeddings ndarray
        """
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model(data)

        return embeddings.detach().cpu().numpy()

    def _run_faiss(self, database: np.ndarray, queries: np.ndarray, 
                    k: int = 100) -> np.ndarray:
        # database shape (N_samples, embedding_dim)
        # queries shape (N_queries, embedding_dim)

        index = faiss.IndexFlatL2(self.embedding_dim) 
        index.add(database)
        D, I = index.search(queries, k)

        return I

    
    def do_inference(self, ) -> pd.DataFrame:
        # preprocess data
        hum_data = self._preprocess_hum_freq()
        song_ids, song_freq = self._preprocess_song_freq()
        print(f'song_ids shape {len(song_ids)}')
        print(f'song_freq shape {len(song_freq)}')

        print(f'hum len {len(hum_data)}')
        # constructing the database
        song_freq = [torch.tensor(x, dtype = torch.float32).unsqueeze(0).unsqueeze(0) for x in song_freq]
        song_tensor = torch.cat(song_freq, dim=0)
        song_embeddings = self._get_embeddings(song_tensor)

        print(f'song_embeddings shape {song_embeddings.shape}')
        # constructing the queries
        query_names = list(hum_data.keys())
        query_freq = [hum_data[qname] for qname in query_names]
        query_freq = [torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0) for x in query_freq]
        query_tensor = torch.cat(query_freq, dim = 0)
        query_embeddings = self._get_embeddings(query_tensor)
        print(f'query_embeddings shape {query_embeddings.shape}')
        neighbors = self._run_faiss(database = song_embeddings,
                                    queries = query_embeddings, k = 10)
        
        
        # use the neighbors to vote the results
        results = {}
        print(f'neighbor shape {neighbors.shape}')
        for qname, nb in zip(query_names, neighbors):
            # chosen = []
            song_id_neighbors = [song_ids[i] for i in nb]
            # ids, counts = np.unique(song_id_neighbors, return_counts = True)
            # sorted_idx = np.argsort(counts)
            # for i in sorted_idx:
            #     if counts[i] == 1:
            #         break
            #     else:
            #         chosen.append(ids[i])

            # if len(chosen) >= 10:
            #     results[qname] = chosen[:10]
            # else:
            #     # if after voting not enough candidates
            #     j = 0
            #     while len(chosen) < 10:
            #         if nb[j] not in chosen:
            #             chosen.append(song_id_neighbors[j])
            #         j+=1
            #     results[qname] = chosen
            results[qname] = song_id_neighbors
        return results          

if __name__ == '__main__':
    path_to_model = r'C:\Users\ASUS\Desktop\hum\data\model_epoch2500.pt'
    val_song_freq_path = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_song_freq.pkl'
    val_hum_freq_path = r'C:\Users\ASUS\Desktop\hum\data\crepe_freq\crepe_freq\val_hum_freq.pkl'

    inferencer = Inferencer(path_to_model, val_song_freq_path, val_hum_freq_path,
                        1100, 512)
    results = inferencer.do_inference()