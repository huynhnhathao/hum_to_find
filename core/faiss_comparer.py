from typing import List
import faiss
from torch import Tensor
import numpy  as np

import arguments as args 

class FaissEvaluator:
    def __init__(self, d: int, song_embeddings: np.ndarray, hum_embeddings: np.ndarray,
            song_labels: List[int], hum_labels: List[int], k=10):
        """Using Faiss to run evaluator on song and hum embeddings
        """
        self.d = d
        self.song_embeddings = song_embeddings
        self.hum_embeddings = hum_embeddings
        self.song_labels = song_labels
        self.hum_labels = hum_labels
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
            if music_id not in neighbors.keys():
                neighbors[music_id] = []

            neighbors[music_id].extend([self.song_labels[x] for x in nb])

        mrr = []
        # voting here
        for key, value in neighbors:
            this_neighbors = []
            song_ids, counts = np.unique(value, return_counts = True)
            sorted_args = np.argsort(counts)
            for nb in reversed(sorted_args):
                this_neighbors.append(int(song_ids[nb]))

            this_neighbors = this_neighbors[:10]
            if key not in this_neighbors:
                mrr.append(0)
            else:
                idx = np.where(np.array(this_neighbors) == int(key),)[0][0] + 1
                mrr.append(1/idx)

        return np.mean(mrr)


if __name__ == '__main__':
    # sanity check here
    song_embeddings = [np.random.randn(4, 64).astype('float32') for i in range(32)]
    queries = [song_embeddings[0]]
    labels = [np.random.randint(0, 10, 4) for i in range(32)]
    evalutator = FaissEvaluator(64, song_embeddings, queries, labels, )
    print(evalutator.evaluate())



#TEST THIS SHIT