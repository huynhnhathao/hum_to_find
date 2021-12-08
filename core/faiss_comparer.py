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
        rr = []
        I = self._run_faiss()
        for i, preds in enumerate(I):
            if i not in preds:
                rr.append(0)
            else:
                idx = np.where(preds==i)[0][0] + 1
                rr.append(1/idx)

        return np.mean(rr)


if __name__ == '__main__':
    # sanity check here
    song_embeddings = [np.random.randn(4, 64).astype('float32') for i in range(32)]
    queries = [song_embeddings[0]]
    labels = [np.random.randint(0, 10, 4) for i in range(32)]
    evalutator = FaissEvaluator(64, song_embeddings, queries, labels, )
    print(evalutator.evaluate())



