"""
HNSW + Product Quantization, combined.

Plain HNSW (hnsw.py) gives fast approximate search but still stores full
float32 vectors, memory doesn't shrink. Plain PQ (pq.py) shrinks memory
but PQFlatIndex still checks every vector (no graph shortcut), so it
doesn't get faster as n grows the way HNSW does. This file is what real
systems like Faiss's IVFPQ do: use HNSW's graph for the O(log n) traversal
trick, but store PQ codes (a few bytes) at each node instead of the full
vector (hundreds of bytes), so it's small AND fast.

The one real design wrinkle: HNSW's original distance function was just
"pre-normalized dot product between two float vectors, cheap and exact."
Once vectors are compressed to codes, there's no float vector sitting in
storage to dot-product against, distance has to go through PQ's asymmetric
distance computation (ADC) instead: build one distance table per query
(query stays a real float vector, it's a one-off, no need to compress it),
then every candidate node's distance is a handful of table lookups. This
file reuses that same table for an entire graph search or insertion pass
instead of rebuilding it per comparison, that reuse is what keeps this fast.

Insertion needs one more accommodation: PQ needs to see representative
data before it can build codebooks (`train()`), so this index must be
trained on a sample of vectors before anything gets added, HNSW's plain
version didn't need this two-phase "train then add" split.
"""
import heapq
import math
import numpy as np
import random

from .pq import ProductQuantizer


def _default_ef_search(n, M):
    return max(30, int(M * math.log2(max(n, 2))))


class HNSWPQIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 pq_m: int = 16, pq_k: int = 256, seed: int = 42,
                 initial_capacity: int = 1024):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2
        self.ef_construction = ef_construction
        self.rng = random.Random(seed)
        self._level_mult = 1.0 / np.log(M)

        self.pq = ProductQuantizer(dim, m=pq_m, k=pq_k, seed=seed)
        self._trained = False

        self._capacity = initial_capacity
        self._codes = np.zeros((self._capacity, pq_m), dtype=np.uint8)  # row i = internal idx i
        self._size = 0
        self._id_to_idx = {}
        self._idx_to_id = []

        self.graph = []
        self.entry_point = None
        self.max_layer = -1

    def train(self, training_vectors):
        """Must be called once, before any add(), with a representative
        sample of the data (doesn't need to be the whole dataset, a few
        thousand vectors is plenty for the codebooks to learn real structure)."""
        self.pq.train(np.asarray(training_vectors, dtype=np.float32))
        self._trained = True

    def _random_layer(self):
        return int(-np.log(self.rng.random()) * self._level_mult)

    def _grow(self):
        new_capacity = self._capacity * 2
        new_codes = np.zeros((new_capacity, self.pq.m), dtype=np.uint8)
        new_codes[:self._capacity] = self._codes
        self._codes = new_codes
        self._capacity = new_capacity

    def _batch_distances(self, idx_list, table):
        """Distance from whatever vector `table` was built for, to every
        idx in idx_list, via PQ's asymmetric distance computation. `table`
        is built ONCE per query/insertion (see search()/add() below) and
        reused across every graph hop, not rebuilt per comparison."""
        if not idx_list:
            return np.array([])
        return self.pq.asymmetric_distances(self._codes[idx_list], table)

    def _search_layer(self, table, entry_points, ef, layer):
        visited = set(entry_points)
        entry_list = list(entry_points)
        entry_dists = self._batch_distances(entry_list, table)

        candidates = list(zip(entry_dists.tolist(), entry_list))
        heapq.heapify(candidates)
        results = [(-d, i) for d, i in candidates]
        heapq.heapify(results)

        while candidates:
            dist, current = heapq.heappop(candidates)
            worst_found = -results[0][0]
            if dist > worst_found and len(results) >= ef:
                break

            unvisited = [n for n in self.graph[current].get(layer, ()) if n not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)

            dists = self._batch_distances(unvisited, table)
            worst_found = -results[0][0]
            for d, neighbor in zip(dists.tolist(), unvisited):
                if d < worst_found or len(results) < ef:
                    heapq.heappush(candidates, (d, neighbor))
                    heapq.heappush(results, (-d, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst_found = -results[0][0]

        return sorted([(-d, i) for d, i in results])

    def add(self, vector_id, vector):
        assert self._trained, "call train() first with a sample of vectors before adding anything"
        vector = np.asarray(vector, dtype=np.float32)
        assert vector.shape == (self.dim,)

        if self._size >= self._capacity:
            self._grow()
        idx = self._size
        self._codes[idx] = self.pq.encode(vector[None, :])[0]  # compress on insert
        self._size += 1
        self._id_to_idx[vector_id] = idx
        self._idx_to_id.append(vector_id)
        self.graph.append({})

        layer = self._random_layer()
        table = self.pq.distance_table(vector)  # built once, reused for this whole insertion

        if self.entry_point is None:
            for l in range(layer + 1):
                self.graph[idx][l] = set()
            self.entry_point = idx
            self.max_layer = layer
            return

        current_ep = self.entry_point
        for l in range(self.max_layer, layer, -1):
            nearest = self._search_layer(table, {current_ep}, ef=1, layer=l)
            current_ep = nearest[0][1]

        eps = {current_ep}
        for l in range(min(layer, self.max_layer), -1, -1):
            found = self._search_layer(table, eps, ef=self.ef_construction, layer=l)
            m = self.M_max0 if l == 0 else self.M
            neighbors = [i for _, i in found[:m]]

            self.graph[idx][l] = set(neighbors)
            for n in neighbors:
                self.graph[n].setdefault(l, set()).add(idx)
                self._prune(n, l, m)

            eps = {i for _, i in found}

        if layer > self.max_layer:
            self.entry_point = idx
            self.max_layer = layer

    def _prune(self, node_idx, layer, m):
        neighbors = self.graph[node_idx][layer]
        if len(neighbors) <= m:
            return
        neighbor_list = list(neighbors)
        # this node has no float vector sitting around anymore, only its
        # code, so reconstruct (decode) it to build a distance table, same
        # pattern as a query, just derived from a stored point instead
        node_vec = self.pq.decode(self._codes[node_idx:node_idx + 1])[0]
        table = self.pq.distance_table(node_vec)
        dists = self._batch_distances(neighbor_list, table)
        order = np.argsort(dists)
        keep = set(neighbor_list[i] for i in order[:m])
        for dropped in neighbors - keep:
            self.graph[dropped][layer].discard(node_idx)
        self.graph[node_idx][layer] = keep

    def search(self, query, k=10, ef_search=None):
        query = np.asarray(query, dtype=np.float32)
        if self.entry_point is None:
            return [], []
        if ef_search is None:
            ef_search = _default_ef_search(self._size, self.M)

        table = self.pq.distance_table(query)  # built once, reused for the whole query

        current_ep = self.entry_point
        for l in range(self.max_layer, 0, -1):
            nearest = self._search_layer(table, {current_ep}, ef=1, layer=l)
            current_ep = nearest[0][1]

        found = self._search_layer(table, {current_ep}, ef=max(ef_search, k), layer=0)
        found = found[:k]
        ids = [self._idx_to_id[i] for _, i in found]
        dists = [d for d, _ in found]
        return ids, dists

    def memory_bytes(self):
        return self._codes[:self._size].nbytes + self.pq.codebooks.nbytes

    def __len__(self):
        return self._size
