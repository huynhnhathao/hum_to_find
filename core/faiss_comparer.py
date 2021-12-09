from typing import List
import faiss
from torch import Tensor
import numpy  as np

import arguments as args 

class FaissEvaluator:
    def __init__(self, d: int, song_embeddings: np.ndarray, hum_embeddings: np.ndarray,
            song_labels: List[int], hum_labels: List[int], k=5):
        """Using Faiss to run evaluator on song and hum embeddings
        """
        self.d = d
        self.song_embeddings = song_embeddings
        self.hum_embeddings = hum_embeddings
        self.song_labels = list(song_labels.flatten())
        self.hum_labels = list(hum_labels.flatten())
        self.k = k

    def _run_faiss(self) -> float:

        index = faiss.IndexFlatL2(self.d) 
        index.add(self.song_embeddings)
        
        D, I = index.search(self.hum_embeddings, self.k)
        return I 

    def evaluate(self) -> float:
        """Compute mrr
        """
        neighbors = {}
        I = self._run_faiss()
        for music_id, nb in zip(self.hum_labels, I):

            if int(music_id) not in neighbors.keys():
                neighbors[int(music_id)] = []

            neighbors[music_id].extend([self.song_labels[x] for x in nb])
        mrr = []
        # voting here
        for key, value in neighbors.items():
            # print(value)
            # rearange the value such that nearest neighbors come first
            value = [value[x:x+5] for x in range(0, len(value), 5)]
            value = [item for sublist in value for item in sublist]


            this_neighbors = []
            song_ids, counts = np.unique(value, return_counts = True)
            sorted_args = np.argsort(counts)

            for nb in reversed(sorted_args):
                if counts[nb] ==1:
                    break
                this_neighbors.append(int(song_ids[nb]))
            j=0
            while len(this_neighbors) < 10 and j < len(value):
                if value[j] not in this_neighbors:
                  this_neighbors.append(value[j])

                j+= 1

            while len(this_neighbors) < 10:
                this_neighbors.append(0)

            this_neighbors = this_neighbors[:10]
            if key not in this_neighbors:
                mrr.append(0)
            else:
                idx = np.where(np.array(this_neighbors) == int(key),)[0][0] + 1
                mrr.append(1/idx)

        return np.mean(mrr)

if __name__ == '__main__':
    # sanity check here
    d = 64
    song_embeddings = np.random.randn(100, d).astype('float32')
    queries = np.array([song_embeddings[x, :] for x in range(5)])
    print(song_embeddings.shape)
    print(queries.shape)
    song_labels = np.arange(0, 100)
    hum_labels = song_labels[:5].reshape(-1, 1)

    evalutator = FaissEvaluator(d, song_embeddings, queries, song_labels, hum_labels )
    print(evalutator.evaluate())

