"""
Brute force nearest neighbor search.

This is our ground truth. Every fancier index we build later (HNSW, PQ)
gets checked against this for correctness. It's also our speed baseline,
everything later is judged by how much faster it is than this.
"""
import numpy as np


class BruteForceIndex:
    def __init__(self, dim: int, metric: str = "cosine"):
        assert metric in ("cosine", "l2"), "metric must be 'cosine' or 'l2'"
        self.dim = dim
        self.metric = metric
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.ids = np.zeros((0,), dtype=np.int64)

    def add(self, vectors: np.ndarray, ids: np.ndarray = None):
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.ndim == 2 and vectors.shape[1] == self.dim, \
            f"expected shape (n, {self.dim}), got {vectors.shape}"

        if ids is None:
            start = len(self.ids)
            ids = np.arange(start, start + len(vectors))
        else:
            ids = np.asarray(ids, dtype=np.int64)

        self.vectors = np.vstack([self.vectors, vectors])
        self.ids = np.concatenate([self.ids, ids])

    def search(self, query: np.ndarray, k: int = 10):
        """Return (ids, distances) of the k closest vectors to query."""
        query = np.asarray(query, dtype=np.float32)
        assert query.shape == (self.dim,), f"expected shape ({self.dim},), got {query.shape}"

        if len(self.vectors) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

        if self.metric == "l2":
            # squared L2 distance to every vector, in one vectorized shot
            diffs = self.vectors - query
            dists = np.einsum("ij,ij->i", diffs, diffs)
        else:  # cosine -> convert to a "distance" where smaller = more similar
            query_norm = query / (np.linalg.norm(query) + 1e-10)
            vec_norms = self.vectors / (np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10)
            sims = vec_norms @ query_norm
            dists = 1.0 - sims  # so "smaller distance" still means "more similar"

        k = min(k, len(dists))
        # argpartition is O(n) instead of a full O(n log n) sort, we only
        # need the top k, not a full ranking of a billion points
        top_k_idx = np.argpartition(dists, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]  # sort just those k

        return self.ids[top_k_idx], dists[top_k_idx]

    def __len__(self):
        return len(self.vectors)
