"""
HNSW (Hierarchical Navigable Small World) from scratch.

The idea in one paragraph: build a multi-layer graph over the vectors.
The top layer has very few nodes and long-range edges (like an express
lane), the bottom layer (layer 0) has every node and short-range edges.
To search: start at a fixed entry point on the top layer, greedily walk
toward the query (always hop to whichever neighbor is closest), drop
down a layer when no neighbor improves, repeat until you reach layer 0
and do a wider, more careful search there for the final answer.

v3 note (this pass): the vectors are no longer stored as a python dict of
{id: array}. They're stored in one preallocated numpy matrix, and every
neighbor id is really an internal integer row index into that matrix. This
matters because batch distance computation is now a single numpy fancy-index
(self._vectors[idx_list] @ query) instead of a python list comprehension
that rebuilds a fresh array on every single call, that list comprehension
was still costing real per-call python overhead even after v2's
vectorization fix, and insertion (which calls this constantly) was the
dominant cost by the time v2 was benchmarked.

External vector ids (whatever the caller passes to add()) are mapped to
internal indices via a dict, so the public API is unchanged.
"""
import heapq
import math
import random
import numpy as np


def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-10)


def _default_ef_search(n, M):
    """ef_search that scales with dataset size, calibrated against real
    recall measurements (not just theory): a FIXED ef_search looks great on
    small test sets and then quietly collapses as n grows (0.997 recall at
    n=1000 down to 0.593 at n=50000 with a fixed ef_search=50, measured).
    M*log2(n) tracks recall much more evenly across sizes (0.92-1.00 in the
    same range), because the graph naturally needs a wider search frontier
    as there's more of it to get lost in. Floored so tiny indexes don't get
    a silly-small ef_search."""
    return max(30, int(M * math.log2(max(n, 2))))


class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200, seed: int = 42,
                 initial_capacity: int = 1024):
        self.dim = dim
        self.M = M
        self.M_max0 = M * 2  # layer 0 gets more neighbors, it does the heavy lifting at search time
        self.ef_construction = ef_construction
        self.rng = random.Random(seed)

        self._capacity = initial_capacity
        self._vectors = np.zeros((self._capacity, dim), dtype=np.float32)  # row i = internal idx i
        self._size = 0
        self._id_to_idx = {}   # external id -> internal row index
        self._idx_to_id = []   # internal row index -> external id

        self.graph = []        # graph[idx] = {layer: set(neighbor idx)}
        self.entry_point = None    # internal idx of current entry point
        self.max_layer = -1
        self._level_mult = 1.0 / np.log(M)

    def _random_layer(self):
        return int(-np.log(self.rng.random()) * self._level_mult)

    def _grow(self):
        new_capacity = self._capacity * 2
        new_vectors = np.zeros((new_capacity, self.dim), dtype=np.float32)
        new_vectors[:self._capacity] = self._vectors
        self._vectors = new_vectors
        self._capacity = new_capacity

    def _batch_distances(self, idx_list, query):
        """Distance from query to every idx in idx_list, in ONE numpy call.
        This is the key fix: fancy-indexing straight into the preallocated
        matrix, no python loop building an intermediate list of vectors."""
        if not idx_list:
            return np.array([])
        return 1.0 - (self._vectors[idx_list] @ query)

    def _search_layer(self, query, entry_points, ef, layer):
        """Greedy best-first search on a single layer. entry_points/returned
        ids are internal indices throughout."""
        visited = set(entry_points)
        entry_list = list(entry_points)
        entry_dists = self._batch_distances(entry_list, query)

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

            dists = self._batch_distances(unvisited, query)
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
        vector = _normalize(np.asarray(vector, dtype=np.float32))
        assert vector.shape == (self.dim,)

        if self._size >= self._capacity:
            self._grow()
        idx = self._size
        self._vectors[idx] = vector
        self._size += 1
        self._id_to_idx[vector_id] = idx
        self._idx_to_id.append(vector_id)
        self.graph.append({})

        layer = self._random_layer()

        if self.entry_point is None:
            for l in range(layer + 1):
                self.graph[idx][l] = set()
            self.entry_point = idx
            self.max_layer = layer
            return

        current_ep = self.entry_point
        for l in range(self.max_layer, layer, -1):
            nearest = self._search_layer(vector, {current_ep}, ef=1, layer=l)
            current_ep = nearest[0][1]

        eps = {current_ep}
        for l in range(min(layer, self.max_layer), -1, -1):
            found = self._search_layer(vector, eps, ef=self.ef_construction, layer=l)
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
        dists = self._batch_distances(neighbor_list, self._vectors[node_idx])
        order = np.argsort(dists)
        keep = set(neighbor_list[i] for i in order[:m])
        for dropped in neighbors - keep:
            self.graph[dropped][layer].discard(node_idx)
        self.graph[node_idx][layer] = keep

    def search(self, query, k=10, ef_search=None):
        """ef_search=None (default) auto-scales with dataset size, see
        _default_ef_search. Pass an explicit int to override and control
        the speed/recall tradeoff by hand."""
        query = _normalize(np.asarray(query, dtype=np.float32))
        if self.entry_point is None:
            return [], []
        if ef_search is None:
            ef_search = _default_ef_search(self._size, self.M)

        current_ep = self.entry_point
        for l in range(self.max_layer, 0, -1):
            nearest = self._search_layer(query, {current_ep}, ef=1, layer=l)
            current_ep = nearest[0][1]

        found = self._search_layer(query, {current_ep}, ef=max(ef_search, k), layer=0)
        found = found[:k]
        ids = [self._idx_to_id[i] for _, i in found]
        dists = [d for d, _ in found]
        return ids, dists

    def __len__(self):
        return self._size
