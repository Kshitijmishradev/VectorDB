"""
Product Quantization (PQ) from scratch.

The problem this solves: brute force and HNSW both still store every
vector as full float32 numbers. At dim=64 that's 256 bytes/vector, at
dim=512 (typical real embedding size) it's 2KB/vector. A billion vectors
at 2KB each is 2 terabytes, nowhere near fitting in RAM. PQ compresses
each vector down to a handful of bytes.

The idea in one paragraph, concrete first: take a 64-dim vector, split it
into 8 chunks of 8 dims each. For each of those 8 "subspaces", run k-means
across the whole dataset to find, say, 256 representative sub-vectors
(centroids). Now any vector's chunk can be replaced by "which of the 256
centroids is it closest to", a single byte (0-255) instead of 8 floats
(32 bytes). Do this for all 8 chunks: a 256-byte vector becomes 8 bytes,
a 32x compression. You've traded exact storage for "the closest of 256
representative options per chunk", which is why this is lossy/approximate,
same tradeoff spirit as HNSW, less memory for a small accuracy cost.

Distance computation on compressed vectors uses "asymmetric distance
computation" (ADC): the query stays as a real float vector (queries are
one-off, no need to compress them), but the stored database vectors are
compressed codes. For each query, precompute a small distance table
(one entry per centroid per subspace) once, then any stored vector's
distance is just a handful of table lookups and additions, no decompression
needed. This is what makes PQ fast, not just small.
"""
import numpy as np


class ProductQuantizer:
    def __init__(self, dim: int, m: int = 16, k: int = 256, seed: int = 0):
        """
        dim: original vector dimensionality
        m: number of subspaces (chunks) to split each vector into. dim must be divisible by m.
           THIS IS A REAL RECALL/COMPRESSION DIAL, not a minor detail, measured on dim=64,
           n=3000 random vectors: m=8 -> 32x compression but recall@10=0.39, m=16 -> 16x
           compression, recall@10=0.68, m=32 -> 8x compression, recall@10=0.89. More
           subspaces means smaller, more precise chunks (less lossy per chunk) at the cost
           of needing more bytes per vector. Default here (16) favors recall being usable
           over maximum compression; tune down for more compression if recall can afford it.
        k: number of centroids per subspace. 256 is the standard choice because
           a centroid index then fits in exactly one byte (uint8), which is where
           the compression ratio (dim*4 / m bytes) actually comes from.
        """
        assert dim % m == 0, f"dim ({dim}) must be divisible by m ({m})"
        self.dim = dim
        self.m = m
        self.k = k
        self.sub_dim = dim // m
        self.rng = np.random.default_rng(seed)
        self.codebooks = None  # shape (m, k, sub_dim) once trained

    def train(self, vectors: np.ndarray, n_iters: int = 15):
        """Learn the codebooks: run k-means independently on each of the m
        subspaces. vectors: shape (n, dim)."""
        vectors = np.asarray(vectors, dtype=np.float32)
        n = len(vectors)
        assert n >= self.k, f"need at least k={self.k} training vectors, got {n}"

        subvectors = vectors.reshape(n, self.m, self.sub_dim)  # (n, m, sub_dim)
        self.codebooks = np.zeros((self.m, self.k, self.sub_dim), dtype=np.float32)

        for subspace in range(self.m):
            data = subvectors[:, subspace, :]  # (n, sub_dim), this subspace's data across all vectors
            self.codebooks[subspace] = self._kmeans(data, self.k, n_iters)

    def _kmeans(self, data, k, n_iters,
                max_distance_bytes=256 * 1024 * 1024, tol=1e-4):
        """Plain Lloyd's algorithm k-means, numpy only, no sklearn.
        data: (n, sub_dim). Returns centroids: (k, sub_dim).

        Called m times per train() (once per subspace) * n_iters each, so
        at real training-set sizes this runs hundreds of times, found (via
        the IVF+PQ large-scale benchmark, where this was the actual
        bottleneck: 74s vs 3.6s for IVF's own centroid k-means on the same
        n) to be spending most of its time on two things a smarter
        formulation avoids:
        1. The (n, k, sub_dim) broadcast distance array, replaced with the
           expanded identity |a-b|^2 = |a|^2 - 2a.b + |b|^2, which only
           needs an (n, k) matrix (one matmul for the cross term). sub_dim
           is small here so this was never a memory blow-up like IVF's
           full-vector clustering was, but it's still real allocation and
           compute overhead repeated hundreds of times.
        2. The python for-loop over k clusters to recompute centroid means,
           replaced with np.add.at doing the same accumulation in one
           vectorized pass instead of k masked-mean operations."""
        n = len(data)
        # init: pick k random points as starting centroids
        centroid_idx = self.rng.choice(n, size=k, replace=False)
        centroids = data[centroid_idx].copy()
        data_sq = (data ** 2).sum(axis=1)  # (n,), computed once, reused every iteration
        batch_size = max(
            1, min(n, max_distance_bytes // (np.dtype(np.float32).itemsize * k)))

        for _ in range(n_iters):
            centroids_sq = (centroids ** 2).sum(axis=1)  # (k,)
            assignments = np.empty(n, dtype=np.int64)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                dists = data[start:end] @ centroids.T
                dists *= -2.0
                dists += data_sq[start:end, None]
                dists += centroids_sq[None, :]
                assignments[start:end] = dists.argmin(axis=1)

            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.float64)
            np.add.at(new_centroids, assignments, data)
            np.add.at(counts, assignments, 1)

            empty = counts == 0
            counts[empty] = 1
            new_centroids /= counts[:, None]
            new_centroids[empty] = centroids[empty]  # empty cluster, keep old position
            shift = np.abs(new_centroids - centroids).max()
            centroids = new_centroids
            if shift < tol:
                break

        return centroids

    def encode(self, vectors: np.ndarray, batch_size: int = 20000) -> np.ndarray:
        """Compress float vectors to uint8 PQ codes.

        Encoding is deliberately chunked.  The old broadcast expression built
        an ``(n, k, sub_dim)`` temporary for every subspace; at ingestion scale
        that both consumed hundreds of MB and repeated work that BLAS can do
        much faster.  The squared-distance identity keeps the largest
        temporary at ``(batch_size, k)`` and turns the expensive part into a
        matrix multiplication.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with shape (n, {self.dim}), got {vectors.shape}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        n = len(vectors)
        subvectors = vectors.reshape(n, self.m, self.sub_dim)
        codes = np.zeros((n, self.m), dtype=np.uint8)

        for subspace in range(self.m):
            data = subvectors[:, subspace, :]  # (n, sub_dim)
            centroids = self.codebooks[subspace]  # (k, sub_dim)
            centroids_sq = (centroids ** 2).sum(axis=1)
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                chunk = data[start:end]
                dists = -2.0 * (chunk @ centroids.T)
                dists += (chunk ** 2).sum(axis=1)[:, None]
                dists += centroids_sq[None, :]
                codes[start:end, subspace] = dists.argmin(axis=1)

        return codes

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decompress: uint8 codes -> approximate float vectors.
        Vectorized: self.codebooks[idx, codes] uses numpy fancy-indexing
        broadcasting (idx is (m,), codes is (n,m)) to pull all n*m centroid
        lookups in one call instead of looping over subspaces in python."""
        n = len(codes)
        idx = np.arange(self.m)
        out = self.codebooks[idx, codes]  # (n, m, sub_dim)
        return out.reshape(n, self.dim)

    def distance_table(self, query: np.ndarray) -> np.ndarray:
        """The ADC trick: precompute distance from query to every centroid,
        in every subspace, once per query. Shape (m, k)."""
        query_sub = query.reshape(self.m, self.sub_dim)
        table = np.zeros((self.m, self.k), dtype=np.float32)
        for subspace in range(self.m):
            diffs = self.codebooks[subspace] - query_sub[subspace]  # (k, sub_dim)
            table[subspace] = (diffs ** 2).sum(axis=1)
        return table

    def asymmetric_distances(self, codes: np.ndarray, table: np.ndarray) -> np.ndarray:
        """Approximate squared L2 distance from the query (already turned into
        `table` via distance_table) to every compressed vector in `codes`.
        This is just m table lookups + a sum, per vector, no decompression.

        Vectorized: this was originally a python for loop over subspaces
        (`for subspace in range(m): dists += table[subspace, codes[:,
        subspace]]`), which is exactly the kind of per-call python overhead
        that turned out to be the dominant cost when this got combined with
        HNSW (61s build at n=10000, 4.5x slower than plain HNSW's 14s, this
        loop was why). table[idx, codes] does the same m*n lookups as one
        numpy fancy-index call instead of m separate python-level ones."""
        n, m = codes.shape
        idx = np.arange(m)
        contributions = table[idx, codes]  # (n, m), broadcasts idx against codes' m axis
        return contributions.sum(axis=1)

    def compression_ratio(self):
        original_bytes = self.dim * 4  # float32
        compressed_bytes = self.m  # one uint8 per subspace
        return original_bytes / compressed_bytes


class PQFlatIndex:
    """Brute-force search, but over PQ-compressed vectors instead of full
    floats. 'Flat' because it still checks every stored vector (no graph
    trick), the speedup here is a memory one, not the O(log n) trick HNSW
    uses, this and HNSW solve different problems and are meant to be
    combined in a real system (compress the vectors HNSW's graph points at),
    not to replace each other."""

    def __init__(self, dim: int, m: int = 16, k: int = 256, seed: int = 0):
        self.pq = ProductQuantizer(dim, m=m, k=k, seed=seed)
        self.codes = np.zeros((0, m), dtype=np.uint8)
        self.ids = np.zeros((0,), dtype=np.int64)
        self._trained = False

    def train(self, training_vectors):
        self.pq.train(training_vectors)
        self._trained = True

    def add(self, vectors, ids=None):
        assert self._trained, "call train() first, PQ needs to learn its codebooks before encoding anything"
        vectors = np.asarray(vectors, dtype=np.float32)
        new_codes = self.pq.encode(vectors)
        if ids is None:
            start = len(self.ids)
            ids = np.arange(start, start + len(vectors))
        self.codes = np.vstack([self.codes, new_codes])
        self.ids = np.concatenate([self.ids, np.asarray(ids, dtype=np.int64)])

    def search(self, query, k=10):
        query = np.asarray(query, dtype=np.float32)
        table = self.pq.distance_table(query)
        dists = self.pq.asymmetric_distances(self.codes, table)

        k = min(k, len(dists))
        top_k_idx = np.argpartition(dists, k - 1)[:k]
        top_k_idx = top_k_idx[np.argsort(dists[top_k_idx])]
        return self.ids[top_k_idx], dists[top_k_idx]

    def memory_bytes(self):
        return self.codes.nbytes + self.pq.codebooks.nbytes

    def __len__(self):
        return len(self.codes)
