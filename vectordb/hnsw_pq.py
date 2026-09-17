"""Memory-efficient HNSW over PQ-compressed vectors.

The graph and PQ codes are resident because HNSW performs small random
accesses while traversing it. Original float32 vectors are optional and,
when enabled, live in a memory-mapped sidecar used only to rerank the final
shortlist. Graph adjacency uses compact integer arrays instead of one Python
``dict``/``set`` tree per node.
"""
import heapq
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

from .pq import ProductQuantizer


STORAGE_VERSION = 1


def _default_ef_search(n, M):
    return max(30, int(M * math.log2(max(n, 2))))


def _tuple_tree(value):
    """Turn the JSON representation of random.Random state back into tuples."""
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


class HNSWPQIndex:
    """HNSW graph whose distance evaluations use PQ asymmetric distance.

    Level zero is a dense fixed-degree matrix because every node owns a row.
    Upper layers are compact: only nodes that actually reach a layer receive a
    row there. With HNSW's geometric level distribution this avoids allocating
    a mostly-empty ``(n, M)`` matrix for every upper layer.
    """

    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200,
                 pq_m: int = 16, pq_k: int = 256, seed: int = 42,
                 initial_capacity: int = 1024, storage_dir: str = None,
                 store_full_vectors: bool = False):
        if dim <= 0 or M <= 1 or ef_construction <= 0:
            raise ValueError("dim, M, and ef_construction must be positive (M > 1)")
        if initial_capacity <= 0:
            raise ValueError("initial_capacity must be positive")
        if store_full_vectors and storage_dir is None:
            raise ValueError("store_full_vectors=True requires storage_dir")

        self.dim = int(dim)
        self.M = int(M)
        self.M_max0 = self.M * 2
        self.ef_construction = int(ef_construction)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self._level_mult = 1.0 / math.log(self.M)
        self.storage_dir = os.fspath(storage_dir) if storage_dir is not None else None
        self.store_full_vectors = bool(store_full_vectors)

        self.pq = ProductQuantizer(self.dim, m=pq_m, k=pq_k, seed=self.seed)
        self._symmetric_tables = None
        self._trained = False
        self._capacity = int(initial_capacity)
        self._size = 0

        self._codes = np.zeros((self._capacity, pq_m), dtype=np.uint8)
        self._ids = np.full(self._capacity, -1, dtype=np.int64)
        self._levels = np.full(self._capacity, -1, dtype=np.int16)
        self._neighbors0 = np.full(
            (self._capacity, self.M_max0), -1, dtype=np.int32)
        self._counts0 = np.zeros(self._capacity, dtype=np.uint16)

        self._upper_neighbors = {}
        self._upper_counts = {}
        self._upper_nodes = {}
        self._upper_rows = {}
        self._upper_sizes = {}
        self._upper_capacities = {}

        self.entry_point = None
        self.max_layer = -1
        self._raw_vectors = None
        self._raw_count = 0
        self.last_search_stats = {"nodes_visited": 0, "candidates_returned": 0}

    @property
    def is_trained(self):
        return self._trained

    def _raw_path(self):
        if self.storage_dir is None:
            raise ValueError("this index has no storage_dir")
        return os.path.join(self.storage_dir, "raw_vectors.f32")

    def train(self, training_vectors, n_iters: int = 15,
              pq_train_size: int = None):
        """Train PQ once, before insertion, on a representative sample."""
        if self._size:
            raise ValueError("cannot retrain a non-empty HNSW index")
        vectors = np.asarray(training_vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected training vectors with shape (n, {self.dim}), got {vectors.shape}")
        if pq_train_size is not None:
            if pq_train_size <= 0:
                raise ValueError("pq_train_size must be positive")
            if pq_train_size < len(vectors):
                sample_rng = np.random.default_rng(self.seed)
                selected = sample_rng.choice(len(vectors), pq_train_size, replace=False)
                vectors = vectors[selected]
        self.pq.train(vectors, n_iters=n_iters)
        self._build_symmetric_tables()
        self._trained = True
        if self.store_full_vectors:
            Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
            self._close_raw()
            with open(self._raw_path(), "wb"):
                pass
            self._raw_count = 0

    def _random_layer(self):
        return int(-math.log(self.rng.random()) * self._level_mult)

    def _reserve(self, required):
        if required <= self._capacity:
            return
        new_capacity = self._capacity
        while new_capacity < required:
            new_capacity *= 2

        def grow(array, shape, fill=0):
            expanded = np.full(shape, fill, dtype=array.dtype)
            expanded[:len(array)] = array
            return expanded

        self._codes = grow(self._codes, (new_capacity, self.pq.m), fill=0)
        self._ids = grow(self._ids, (new_capacity,), fill=-1)
        self._levels = grow(self._levels, (new_capacity,), fill=-1)
        self._neighbors0 = grow(
            self._neighbors0, (new_capacity, self.M_max0), fill=-1)
        self._counts0 = grow(self._counts0, (new_capacity,), fill=0)
        self._capacity = new_capacity

    def _ensure_upper_layer(self, layer):
        if layer in self._upper_neighbors:
            return
        capacity = 16
        self._upper_neighbors[layer] = np.full(
            (capacity, self.M), -1, dtype=np.int32)
        self._upper_counts[layer] = np.zeros(capacity, dtype=np.uint16)
        self._upper_nodes[layer] = np.full(capacity, -1, dtype=np.int32)
        self._upper_rows[layer] = {}
        self._upper_sizes[layer] = 0
        self._upper_capacities[layer] = capacity

    def _grow_upper(self, layer):
        old_capacity = self._upper_capacities[layer]
        new_capacity = old_capacity * 2
        neighbors = np.full((new_capacity, self.M), -1, dtype=np.int32)
        neighbors[:old_capacity] = self._upper_neighbors[layer]
        counts = np.zeros(new_capacity, dtype=np.uint16)
        counts[:old_capacity] = self._upper_counts[layer]
        nodes = np.full(new_capacity, -1, dtype=np.int32)
        nodes[:old_capacity] = self._upper_nodes[layer]
        self._upper_neighbors[layer] = neighbors
        self._upper_counts[layer] = counts
        self._upper_nodes[layer] = nodes
        self._upper_capacities[layer] = new_capacity

    def _register_upper_node(self, node, layer):
        self._ensure_upper_layer(layer)
        rows = self._upper_rows[layer]
        if node in rows:
            return rows[node]
        size = self._upper_sizes[layer]
        if size >= self._upper_capacities[layer]:
            self._grow_upper(layer)
        self._upper_nodes[layer][size] = node
        rows[node] = size
        self._upper_sizes[layer] = size + 1
        return size

    def _neighbor_storage(self, node, layer):
        if layer == 0:
            return self._neighbors0[node], self._counts0, node
        row = self._upper_rows.get(layer, {}).get(int(node))
        if row is None:
            return None, None, None
        return self._upper_neighbors[layer][row], self._upper_counts[layer], row

    def _neighbors(self, node, layer):
        values, counts, row = self._neighbor_storage(node, layer)
        if values is None:
            return np.empty(0, dtype=np.int32)
        return values[:int(counts[row])]

    def _replace_neighbors(self, node, layer, neighbors):
        if layer > 0:
            self._register_upper_node(node, layer)
        values, counts, row = self._neighbor_storage(node, layer)
        neighbors = np.asarray(neighbors, dtype=np.int32)
        if len(neighbors) > len(values):
            raise ValueError("neighbor list exceeds layer degree")
        values.fill(-1)
        values[:len(neighbors)] = neighbors
        counts[row] = len(neighbors)

    def _remove_neighbor(self, node, layer, target):
        values, counts, row = self._neighbor_storage(node, layer)
        if values is None:
            return
        count = int(counts[row])
        matches = np.flatnonzero(values[:count] == target)
        if not len(matches):
            return
        position = int(matches[0])
        count -= 1
        if position != count:
            values[position] = values[count]
        values[count] = -1
        counts[row] = count

    def _batch_distances(self, idx_list, table):
        if len(idx_list) == 0:
            return np.empty(0, dtype=np.float32)
        indices = np.asarray(idx_list, dtype=np.int64)
        return self.pq.asymmetric_distances(self._codes[indices], table)

    def _build_symmetric_tables(self):
        """Precompute reconstructed-code distances used by graph pruning.

        The original implementation decoded a node and rebuilt an ADC table
        every time one of its full neighbor lists needed pruning. Both sides
        are already PQ codes, so the same distance is a lookup in a static
        codeword-to-codeword tensor. For m=32 and k=256 this costs 8 MiB once
        and removes thousands of tiny table builds from every insertion batch.
        """
        tables = np.empty((self.pq.m, self.pq.k, self.pq.k), dtype=np.float32)
        for subspace in range(self.pq.m):
            codewords = self.pq.codebooks[subspace]
            differences = codewords[:, None, :] - codewords[None, :, :]
            tables[subspace] = (differences ** 2).sum(axis=2)
        self._symmetric_tables = tables

    def _symmetric_distances(self, node, candidates):
        candidates = np.asarray(candidates, dtype=np.int64)
        subspaces = np.arange(self.pq.m)[None, :]
        node_code = self._codes[node][None, :]
        candidate_codes = self._codes[candidates]
        contributions = self._symmetric_tables[
            subspaces, node_code, candidate_codes]
        return contributions.sum(axis=1)

    def _search_layer(self, table, entry_points, ef, layer):
        entry_list = [int(item) for item in entry_points]
        visited = set(entry_list)
        entry_dists = self._batch_distances(entry_list, table)
        candidates = list(zip(entry_dists.tolist(), entry_list))
        heapq.heapify(candidates)
        results = [(-distance, node) for distance, node in candidates]
        heapq.heapify(results)

        while candidates:
            distance, current = heapq.heappop(candidates)
            worst_found = -results[0][0]
            if distance > worst_found and len(results) >= ef:
                break
            unvisited = [
                int(node) for node in self._neighbors(current, layer)
                if int(node) not in visited]
            if not unvisited:
                continue
            visited.update(unvisited)
            distances = self._batch_distances(unvisited, table)
            worst_found = -results[0][0]
            for candidate_distance, neighbor in zip(distances.tolist(), unvisited):
                if candidate_distance < worst_found or len(results) < ef:
                    heapq.heappush(candidates, (candidate_distance, neighbor))
                    heapq.heappush(results, (-candidate_distance, neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)
                    worst_found = -results[0][0]
        return sorted((-distance, node) for distance, node in results), len(visited)

    def _connect_neighbor(self, node, layer, candidate, max_degree):
        existing = self._neighbors(node, layer).astype(np.int32, copy=True)
        if np.any(existing == candidate):
            return
        proposed = np.append(existing, np.int32(candidate))
        if len(proposed) <= max_degree:
            self._replace_neighbors(node, layer, proposed)
            return

        distances = self._symmetric_distances(node, proposed)
        order = np.argsort(distances, kind="stable")
        keep = proposed[order[:max_degree]]
        dropped = set(proposed.tolist()) - set(keep.tolist())
        self._replace_neighbors(node, layer, keep)
        for dropped_node in dropped:
            self._remove_neighbor(int(dropped_node), layer, node)

    def _insert_preencoded(self, vector_id, vector, code):
        idx = self._size
        self._codes[idx] = code
        self._ids[idx] = int(vector_id)
        layer = self._random_layer()
        if layer > np.iinfo(np.int16).max:
            raise OverflowError("generated HNSW level exceeds int16 storage")
        self._levels[idx] = layer
        for upper_layer in range(1, layer + 1):
            self._register_upper_node(idx, upper_layer)
        self._size += 1

        table = self.pq.distance_table(vector)
        if self.entry_point is None:
            self.entry_point = idx
            self.max_layer = layer
            return

        current_entry = self.entry_point
        for current_layer in range(self.max_layer, layer, -1):
            nearest, _ = self._search_layer(
                table, [current_entry], ef=1, layer=current_layer)
            current_entry = nearest[0][1]

        entry_points = [current_entry]
        for current_layer in range(min(layer, self.max_layer), -1, -1):
            found, _ = self._search_layer(
                table, entry_points, ef=self.ef_construction,
                layer=current_layer)
            max_degree = self.M_max0 if current_layer == 0 else self.M
            selected = [node for _, node in found[:max_degree]]
            self._replace_neighbors(idx, current_layer, selected)
            for neighbor in selected:
                self._connect_neighbor(neighbor, current_layer, idx, max_degree)
            entry_points = [node for _, node in found]

        if layer > self.max_layer:
            self.entry_point = idx
            self.max_layer = layer

    def add_batch(self, vectors, ids, pq_batch_size: int = 20000):
        """PQ-encode in bulk, then perform dependency-ordered graph inserts."""
        if not self._trained:
            raise AssertionError("call train() first with a sample of vectors")
        vectors = np.asarray(vectors, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with shape (n, {self.dim}), got {vectors.shape}")
        if ids.ndim != 1 or len(ids) != len(vectors):
            raise ValueError("ids must be one-dimensional and match vectors")
        if len(vectors) == 0:
            return

        self._reserve(self._size + len(vectors))
        codes = self.pq.encode(vectors, batch_size=pq_batch_size)
        if self.store_full_vectors:
            self._close_raw()
            Path(self.storage_dir).mkdir(parents=True, exist_ok=True)
            with open(self._raw_path(), "ab") as raw_file:
                raw_file.write(np.ascontiguousarray(vectors, dtype="<f4").tobytes())

        for vector_id, vector, code in zip(ids, vectors, codes):
            self._insert_preencoded(int(vector_id), vector, code)
        self._raw_count = self._size if self.store_full_vectors else 0

    def add(self, vector_id, vector):
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dim,):
            raise ValueError(f"expected a flat vector of dim {self.dim}, got {vector.shape}")
        self.add_batch(vector[None, :], np.asarray([vector_id], dtype=np.int64))

    def _map_raw(self):
        if not self.store_full_vectors:
            raise ValueError("this index does not store full vectors")
        if self._raw_vectors is not None:
            return
        expected = self._size * self.dim * np.dtype(np.float32).itemsize
        actual = os.path.getsize(self._raw_path()) if os.path.exists(self._raw_path()) else 0
        if expected != actual:
            raise ValueError(
                f"raw vector file size mismatch: expected {expected}, found {actual}")
        if self._size:
            self._raw_vectors = np.memmap(
                self._raw_path(), dtype="<f4", mode="r",
                shape=(self._size, self.dim))
        else:
            self._raw_vectors = np.empty((0, self.dim), dtype=np.float32)

    def _close_raw(self):
        if isinstance(self._raw_vectors, np.memmap):
            self._raw_vectors._mmap.close()
        self._raw_vectors = None

    def search(self, query, k=10, ef_search=None, rerank=0):
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (self.dim,):
            raise ValueError(f"expected a flat query of dim {self.dim}, got {query.shape}")
        if k <= 0 or rerank < 0:
            raise ValueError("k must be positive and rerank must be non-negative")
        if rerank and not self.store_full_vectors:
            raise ValueError(
                "reranking requires an index created with store_full_vectors=True")
        if self.entry_point is None:
            self.last_search_stats = {"nodes_visited": 0, "candidates_returned": 0}
            return [], []
        if ef_search is None:
            ef_search = _default_ef_search(self._size, self.M)
        if ef_search <= 0:
            raise ValueError("ef_search must be positive")

        shortlist_size = min(self._size, max(k, rerank) if rerank else k)
        effective_ef = min(self._size, max(int(ef_search), shortlist_size))
        table = self.pq.distance_table(query)
        current_entry = self.entry_point
        visited_total = 0
        for layer in range(self.max_layer, 0, -1):
            nearest, visited = self._search_layer(
                table, [current_entry], ef=1, layer=layer)
            current_entry = nearest[0][1]
            visited_total += visited
        found, visited = self._search_layer(
            table, [current_entry], ef=effective_ef, layer=0)
        visited_total += visited
        found = found[:shortlist_size]
        self.last_search_stats = {
            "nodes_visited": int(visited_total),
            "candidates_returned": len(found),
        }
        if not found:
            return [], []

        internal_ids = np.asarray([node for _, node in found], dtype=np.int64)
        result_k = min(k, len(internal_ids))
        if rerank:
            self._map_raw()
            exact_vectors = np.asarray(self._raw_vectors[internal_ids])
            exact_distances = ((exact_vectors - query) ** 2).sum(axis=1)
            order = np.argpartition(exact_distances, result_k - 1)[:result_k]
            order = order[np.argsort(exact_distances[order])]
            return (self._ids[internal_ids[order]].tolist(),
                    exact_distances[order].tolist())

        approximate = np.asarray([distance for distance, _ in found], dtype=np.float32)
        return (self._ids[internal_ids[:result_k]].tolist(),
                approximate[:result_k].tolist())

    def code_bytes(self):
        """Logical bytes for live PQ codes and their codebooks only."""
        codebooks = self.pq.codebooks.nbytes if self.pq.codebooks is not None else 0
        return self._codes[:self._size].nbytes + codebooks

    def graph_bytes(self):
        """Allocated bytes for graph adjacency and compact upper maps."""
        total = self._neighbors0.nbytes + self._counts0.nbytes + self._levels.nbytes
        for layer in self._upper_neighbors:
            total += self._upper_neighbors[layer].nbytes
            total += self._upper_counts[layer].nbytes
            total += self._upper_nodes[layer].nbytes
            rows = self._upper_rows[layer]
            total += sys.getsizeof(rows)
            total += sum(sys.getsizeof(key) + sys.getsizeof(value)
                         for key, value in rows.items())
        return total

    def resident_memory_bytes(self):
        codebooks = self.pq.codebooks.nbytes if self.pq.codebooks is not None else 0
        symmetric = (
            self._symmetric_tables.nbytes if self._symmetric_tables is not None else 0)
        return (self._codes.nbytes + self._ids.nbytes + self.graph_bytes() +
                codebooks + symmetric)

    def memory_bytes(self):
        """Compatibility alias; unlike the old method, includes the graph."""
        return self.resident_memory_bytes()

    def raw_vector_bytes(self):
        if not self.store_full_vectors or self.storage_dir is None:
            return 0
        return os.path.getsize(self._raw_path()) if os.path.exists(self._raw_path()) else 0

    def disk_bytes(self):
        if self.storage_dir is None or not os.path.isdir(self.storage_dir):
            return self.raw_vector_bytes()
        return sum(
            os.path.getsize(os.path.join(root, name))
            for root, _, names in os.walk(self.storage_dir)
            for name in names)

    @staticmethod
    def _atomic_save_array(path, array):
        temporary = path + ".tmp"
        with open(temporary, "wb") as output:
            np.save(output, array, allow_pickle=False)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)

    def save(self, storage_dir: str = None):
        """Persist a reusable graph snapshot; metadata is committed last."""
        if not self._trained:
            raise AssertionError("nothing to save; call train() first")
        destination = os.fspath(storage_dir or self.storage_dir or "")
        if not destination:
            raise ValueError("save requires storage_dir")
        if self.store_full_vectors and self.storage_dir is not None:
            if os.path.abspath(destination) != os.path.abspath(self.storage_dir):
                raise ValueError("an index with a raw sidecar must save to its storage_dir")
        self.storage_dir = destination
        Path(destination).mkdir(parents=True, exist_ok=True)

        arrays = {
            "codes.npy": self._codes[:self._size],
            "ids.npy": self._ids[:self._size],
            "levels.npy": self._levels[:self._size],
            "neighbors0.npy": self._neighbors0[:self._size],
            "counts0.npy": self._counts0[:self._size],
        }
        upper_layers = sorted(self._upper_neighbors)
        for layer in upper_layers:
            size = self._upper_sizes[layer]
            arrays[f"upper_{layer}_nodes.npy"] = self._upper_nodes[layer][:size]
            arrays[f"upper_{layer}_neighbors.npy"] = self._upper_neighbors[layer][:size]
            arrays[f"upper_{layer}_counts.npy"] = self._upper_counts[layer][:size]
        for name, array in arrays.items():
            self._atomic_save_array(os.path.join(destination, name), array)

        metadata_path = os.path.join(destination, "hnsw_meta.npz")
        temporary = metadata_path + ".tmp"
        with open(temporary, "wb") as output:
            np.savez(
                output,
                storage_version=np.int64(STORAGE_VERSION),
                dim=np.int64(self.dim), M=np.int64(self.M),
                ef_construction=np.int64(self.ef_construction),
                pq_m=np.int64(self.pq.m), pq_k=np.int64(self.pq.k),
                seed=np.int64(self.seed), size=np.int64(self._size),
                entry_point=np.int64(-1 if self.entry_point is None else self.entry_point),
                max_layer=np.int64(self.max_layer),
                store_full_vectors=np.bool_(self.store_full_vectors),
                raw_count=np.int64(self._raw_count),
                upper_layers=np.asarray(upper_layers, dtype=np.int16),
                codebooks=self.pq.codebooks,
                rng_state=np.asarray(json.dumps(self.rng.getstate())),
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, metadata_path)
        return destination

    @classmethod
    def load(cls, storage_dir: str):
        storage_dir = os.fspath(storage_dir)
        metadata_path = os.path.join(storage_dir, "hnsw_meta.npz")
        with np.load(metadata_path, allow_pickle=False) as metadata:
            version = int(metadata["storage_version"])
            if version != STORAGE_VERSION:
                raise ValueError(f"unsupported HNSW storage version: {version}")
            size = int(metadata["size"])
            index = cls(
                dim=int(metadata["dim"]), M=int(metadata["M"]),
                ef_construction=int(metadata["ef_construction"]),
                pq_m=int(metadata["pq_m"]), pq_k=int(metadata["pq_k"]),
                seed=int(metadata["seed"]), initial_capacity=max(size, 1),
                storage_dir=storage_dir,
                store_full_vectors=bool(metadata["store_full_vectors"]),
            )
            index.pq.codebooks = np.asarray(metadata["codebooks"], dtype=np.float32)
            index._build_symmetric_tables()
            index._trained = True
            index._size = size
            index.entry_point = int(metadata["entry_point"])
            if index.entry_point < 0:
                index.entry_point = None
            index.max_layer = int(metadata["max_layer"])
            index._raw_count = int(metadata["raw_count"])
            index.rng.setstate(_tuple_tree(json.loads(str(metadata["rng_state"]))))
            upper_layers = [int(layer) for layer in metadata["upper_layers"]]

        def load_array(name, dtype):
            return np.asarray(
                np.load(os.path.join(storage_dir, name), allow_pickle=False),
                dtype=dtype)

        index._codes[:] = load_array("codes.npy", np.uint8)
        index._ids[:] = load_array("ids.npy", np.int64)
        index._levels[:] = load_array("levels.npy", np.int16)
        index._neighbors0[:] = load_array("neighbors0.npy", np.int32)
        index._counts0[:] = load_array("counts0.npy", np.uint16)

        for layer in upper_layers:
            nodes = load_array(f"upper_{layer}_nodes.npy", np.int32)
            neighbors = load_array(f"upper_{layer}_neighbors.npy", np.int32)
            counts = load_array(f"upper_{layer}_counts.npy", np.uint16)
            capacity = max(len(nodes), 16)
            index._upper_neighbors[layer] = np.full(
                (capacity, index.M), -1, dtype=np.int32)
            index._upper_counts[layer] = np.zeros(capacity, dtype=np.uint16)
            index._upper_nodes[layer] = np.full(capacity, -1, dtype=np.int32)
            index._upper_neighbors[layer][:len(nodes)] = neighbors
            index._upper_counts[layer][:len(nodes)] = counts
            index._upper_nodes[layer][:len(nodes)] = nodes
            index._upper_rows[layer] = {
                int(node): row for row, node in enumerate(nodes.tolist())}
            index._upper_sizes[layer] = len(nodes)
            index._upper_capacities[layer] = capacity

        if index.store_full_vectors:
            expected = index._raw_count * index.dim * np.dtype(np.float32).itemsize
            actual = os.path.getsize(index._raw_path()) if os.path.exists(index._raw_path()) else 0
            if expected != actual or index._raw_count != index._size:
                raise ValueError("raw vector sidecar does not match HNSW metadata")
        return index

    def close(self):
        self._close_raw()

    def __len__(self):
        return self._size
