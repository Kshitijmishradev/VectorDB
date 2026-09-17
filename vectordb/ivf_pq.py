"""
IVF + PQ: cluster-routed, disk-backed vector search.

The problem this solves, directly: everything built so far (HNSW, PQ,
HNSW+PQ) keeps the ENTIRE index resident in memory. At a billion vectors,
even PQ-compressed, the graph structure and codes together are still too
much RAM for a laptop. This file fixes that by never needing the whole
dataset in memory at once, only a small routing index.

The idea in one paragraph, concrete first: instead of one graph connecting
every vector to its neighbors, cluster the whole dataset into `nlist`
groups (k-means, same algorithm already written for PQ, just run on full
vectors instead of subvectors). Store only the `nlist` cluster centroids
in memory, if nlist=32000 and dim=64, that's 32000*64*4 bytes = 8MB,
totally negligible even at a billion vectors behind those clusters. Every
actual vector, PQ-encoded to a few bytes, gets written to a small file on
disk, one file per cluster ("inverted list" / "posting list", the classic
name for this). At query time: compare the query against all `nlist`
centroids (cheap, one numpy call, they're all in RAM), find the `nprobe`
closest clusters, and ONLY read those specific files off disk. Everything
else on disk is never touched for that query.

Why this beats a disk-paged HNSW graph for this use case: HNSW's search
pattern is a chain of small random reads (hop to a node, read its
neighbor list, hop again), the worst-case access pattern for disk
latency. IVF's access pattern is a handful of large SEQUENTIAL reads
(read one whole cluster file start to finish), which is what disks (even
spinning ones, and especially SSDs) are actually good at. This is closer
to what real billion-scale systems like Microsoft's SPANN do than raw
disk-backed HNSW is.

The real cost, honestly: IVF is coarser-grained than HNSW's graph search.
If the true nearest neighbor's cluster isn't among the nprobe clusters
probed, it's missed entirely, no amount of searching within the wrong
clusters finds it. `nprobe` is the resulting speed/recall dial, same
spirit as ef_search and PQ's m, probe more clusters for better recall at
the cost of touching more disk.
"""
import os
import shutil
import numpy as np

from .pq import ProductQuantizer


def _kmeans(data, k, n_iters, rng, batch_size=20000, minibatch_size=None,
            tol=1e-4, max_distance_bytes=256 * 1024 * 1024,
            training_mode="legacy"):
    """Same Lloyd's algorithm as ProductQuantizer._kmeans, factored out
    here since IVF needs to cluster full vectors, not PQ's subvectors.

    Fixes vs the original naive version, each found by thinking through
    what happens at real IVF scale (nlist in the thousands to hundreds of
    thousands, training sets scaling right along with it) rather than
    waiting to hit a crash or a multi-hour run mid-benchmark:

    1. Distance computation used to build a (n, k, dim) broadcast array
       (`(data[:, None, :] - centroids[None, :, :]) ** 2`). That's fine
       for PQ's subspaces (k=256, sub_dim small), but here k=nlist can be
       thousands and dim is the FULL vector, so at n=50000, k=4000,
       dim=64 that array alone is 50000*4000*64*4 bytes = 51GB. Rewritten
       using the expanded squared-distance identity
       |a-b|^2 = |a|^2 - 2a.b + |b|^2, which only ever materializes an
       (n, k) matrix (a single matmul for the cross term), the same
       trick real k-means implementations use.
    2. Centroid recomputation used to loop over k clusters in python,
       each iteration masking and averaging over all n points, an O(n*k)
       cost per iteration before you even count the k-loop overhead.
       Replaced with np.add.at, which does the same accumulation in one
       vectorized pass, O(n).
    3. Even after (1), a single (n, k) distance matrix is still O(n*k)
       MEMORY, not just compute, and with several float32 temporaries
       alive at once (the matmul result, the *2.0, the subtraction, the
       addition) that's easily 4-5x the array's own size. At the scale
       this was built for (n=160000 training vectors, nlist=4000), that
       one line was ~13GB and OOM-killed the process outright, found by
       actually trying to run the 1M-vector benchmark, not by guessing.
       Fixed by computing the (n, k) matrix in row-chunks (`batch_size`
       rows at a time): peak memory becomes O(batch_size * k) instead of
       O(n * k), independent of the full training-set size. The row loop
       here does NOT change the algorithm's complexity, it's still
       O(n_iters * n * k) total work, exactly what Lloyd's k-means costs
       everywhere, chunking only bounds how much of that is live in
       memory at once.
    4. That O(n_iters * n * k) cost is exactly what breaks down at real
       IVF scale, though, since both n (train_n) and k (nlist) grow
       together as the dataset grows (nlist ~ 4*sqrt(dataset_size)).
       Projected from measured 1M-scale numbers, training at a
       1B-vector target (nlist ~126,000, train_n ~5M) would cost several
       HOURS on its own, before a single vector gets inserted. Two
       independent fixes for that, both standard in real systems
       (sklearn's MiniBatchKMeans, faiss's clustering):
         - `minibatch_size`: if set, every iteration samples a fresh
           random subset of this many rows and updates centroids from
           only that subset, instead of every training point. This
           makes each iteration's cost O(minibatch_size * k) instead of
           O(train_n * k), independent of how large the training sample
           is. It trades exact Lloyd's convergence for mini-batch
           k-means's approximation, a tradeoff IVF training already
           accepts elsewhere (it trains on a capped sample, not the
           full dataset, to begin with). None (default) preserves exact
           full-batch behavior, used for the small-scale correctness
           tests where this cost was never the bottleneck.
         - `tol`: stop iterating once the largest single centroid's
           movement between iterations drops below this, instead of
           always running all n_iters regardless of whether anything is
           still changing. Most of the real movement happens in the
           first handful of iterations, forcing every iteration to run
           anyway wastes the same kind of work the rest of this project
           has spent five bug-fixes eliminating, just algorithmically
           this time instead of via an unvectorized loop.
    5. Inside the per-chunk distance computation, the three terms of the
       identity are now accumulated in-place (`dists = ...; dists += ...;
       dists += ...`) instead of one compound expression. A compound
       expression holds the matmul result, the scaled copy, and two more
       intermediates alive simultaneously; accumulating in-place reuses
       the same buffer, cutting peak temporary memory for this step
       roughly in half without changing the result.
    """
    if training_mode not in {"legacy", "accumulated"}:
        raise ValueError("training_mode must be 'legacy' or 'accumulated'")
    n = len(data)
    # The distance matrix is float32 with shape (batch, k).  A fixed row
    # batch of 20k is already ~1.5 GiB when k=20k (the 25M-vector benchmark),
    # before BLAS temporaries.  Bound it by bytes as k grows.
    distance_batch_size = max(
        1, min(batch_size, max_distance_bytes // (np.dtype(np.float32).itemsize * k)))
    centroid_idx = rng.choice(n, size=k, replace=False)
    centroids = data[centroid_idx].copy()

    full_batch = minibatch_size is None or minibatch_size >= n
    if full_batch:
        data_sq_full = (data ** 2).sum(axis=1)  # (n,), computed once, reused every iteration

    accumulated_sums = np.zeros((k, data.shape[1]), dtype=np.float64)
    accumulated_counts = np.zeros(k, dtype=np.int64)
    for _ in range(n_iters):
        if full_batch:
            sample = data
            sample_sq = data_sq_full
        else:
            # fresh random subsample each iteration: mini-batch k-means,
            # not full-batch Lloyd's. See point 4 above.
            sample_idx = rng.choice(n, size=minibatch_size, replace=False)
            sample = data[sample_idx]
            sample_sq = (sample ** 2).sum(axis=1)

        m = len(sample)
        centroids_sq = (centroids ** 2).sum(axis=1)  # (k,)
        assignments = np.empty(m, dtype=np.int64)
        for start in range(0, m, distance_batch_size):
            end = min(start + distance_batch_size, m)
            # (batch, k) via one matmul instead of (n, k, dim) via broadcasting,
            # bounded to `batch_size` rows instead of all m at once, and
            # accumulated in-place instead of one compound expression (point 5)
            dists = sample[start:end] @ centroids.T
            dists *= -2.0
            dists += sample_sq[start:end, None]
            dists += centroids_sq[None, :]
            assignments[start:end] = dists.argmin(axis=1)

        batch_sums = np.zeros_like(centroids)
        counts = np.zeros(k, dtype=np.int64)
        np.add.at(batch_sums, assignments, sample)
        np.add.at(counts, assignments, 1)

        if training_mode == "accumulated" and not full_batch:
            accumulated_sums += batch_sums
            accumulated_counts += counts
            new_centroids = centroids.copy()
            occupied = accumulated_counts > 0
            new_centroids[occupied] = (
                accumulated_sums[occupied]
                / accumulated_counts[occupied, None]
            ).astype(np.float32)
            shift = np.abs(new_centroids - centroids).max()
            centroids = new_centroids
            if shift < tol:
                break
            continue

        new_centroids = batch_sums
        empty = counts == 0
        counts[empty] = 1
        new_centroids /= counts[:, None]
        new_centroids[empty] = centroids[empty]  # empty cluster, keep old position

        shift = np.abs(new_centroids - centroids).max()
        centroids = new_centroids
        if shift < tol:
            break  # converged, no point burning the remaining iterations

    return centroids


class IVFPQIndex:
    def __init__(self, dim: int, nlist: int, pq_m: int = 16, pq_k: int = 256,
                 storage_dir: str = "./ivf_storage", seed: int = 0,
                 store_full_vectors: bool = False,
                 routing_backend: str = "numpy",
                 pq_mode: str = "standard"):
        """
        dim: vector dimensionality
        nlist: number of coarse clusters. This is THE memory-vs-recall
               dial: more clusters means each one is smaller (less to scan
               per probe, and a better chance the true nearest neighbor's
               cluster gets probed), but a bigger in-memory centroid table.
               Common real-world default: roughly sqrt(n) to 4*sqrt(n).
        storage_dir: where posting-list files live on disk. Nothing about
               the actual vector data ever lives anywhere else.
        """
        self.dim = dim
        self.nlist = nlist
        self.storage_dir = storage_dir
        self.rng = np.random.default_rng(seed)
        self.store_full_vectors = bool(store_full_vectors)
        if pq_mode not in {"standard", "residual"}:
            raise ValueError("pq_mode must be 'standard' or 'residual'")
        self.pq_mode = pq_mode
        self.coarse_training = "legacy"
        if routing_backend not in {"numpy", "mlx"}:
            raise ValueError("routing_backend must be 'numpy' or 'mlx'")
        self.routing_backend = routing_backend

        self.pq = ProductQuantizer(dim, m=pq_m, k=pq_k, seed=seed)
        self.centroids = None  # (nlist, dim), the ONLY per-vector-scale-independent thing kept in RAM
        self._centroid_norms = None
        self._trained = False
        self._cluster_sizes = np.zeros(nlist, dtype=np.int64)
        self._base_cluster_sizes = np.zeros(nlist, dtype=np.int64)
        self._packed_offsets = None
        self._packed_records = None
        self._raw_count = 0
        self._raw_vectors = None
        self._mlx_router = None

        os.makedirs(storage_dir, exist_ok=True)

    def _cluster_path(self, cluster_id):
        return os.path.join(self.storage_dir, f"cluster_{cluster_id}.bin")

    def _delta_path(self, cluster_id):
        return os.path.join(self.storage_dir, f"delta_{cluster_id}.bin")

    def _packed_path(self):
        return os.path.join(self.storage_dir, "postings.bin")

    def _raw_path(self):
        return os.path.join(self.storage_dir, "raw_vectors.f32")

    @property
    def _record_dtype(self):
        fields = [("id", "<i8"), ("code", "u1", (self.pq.m,))]
        if self.store_full_vectors:
            fields.append(("raw_row", "<i8"))
        return np.dtype(fields)

    @property
    def is_compacted(self) -> bool:
        return self._packed_offsets is not None

    def _close_packed(self):
        if isinstance(self._packed_records, np.memmap):
            self._packed_records._mmap.close()
        self._packed_records = None

    def _map_packed(self):
        self._close_packed()
        total = int(self._packed_offsets[-1])
        if total == 0:
            self._packed_records = np.empty(0, dtype=self._record_dtype)
            return
        expected_bytes = total * self._record_dtype.itemsize
        actual_bytes = os.path.getsize(self._packed_path())
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"packed postings size mismatch: expected {expected_bytes} bytes, "
                f"found {actual_bytes}")
        self._packed_records = np.memmap(
            self._packed_path(), dtype=self._record_dtype, mode="r", shape=(total,))

    def _close_raw(self):
        if isinstance(self._raw_vectors, np.memmap):
            self._raw_vectors._mmap.close()
        self._raw_vectors = None

    def _map_raw(self):
        if not self.store_full_vectors:
            raise ValueError("full vectors were not stored for this index")
        if self._raw_vectors is not None:
            return
        expected_bytes = self._raw_count * self.dim * np.dtype(np.float32).itemsize
        actual_bytes = os.path.getsize(self._raw_path()) if os.path.exists(self._raw_path()) else 0
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"raw vector file size mismatch: expected {expected_bytes}, "
                f"found {actual_bytes}")
        if self._raw_count == 0:
            self._raw_vectors = np.empty((0, self.dim), dtype=np.float32)
        else:
            self._raw_vectors = np.memmap(
                self._raw_path(), dtype="<f4", mode="r",
                shape=(self._raw_count, self.dim))

    @property
    def is_trained(self) -> bool:
        """Public accessor for _trained, added for the API layer (api.py)
        so external code checking training status doesn't have to reach
        into a leading-underscore attribute."""
        return self._trained

    def train(self, training_vectors: np.ndarray, n_iters: int = 15,
              minibatch_size: int = None, tol: float = 1e-4,
              pq_train_size: int = None,
              coarse_training: str = "legacy"):
        """One-time setup: learn both the coarse cluster centroids AND the
        PQ codebooks from a representative sample. Neither needs the full
        dataset, a sample large enough to be representative is enough.

        minibatch_size: passed through to the coarse centroid k-means (see
        _kmeans's docstring). Leave None for exact full-batch behavior at
        small/medium scale; set it (e.g. 20000-50000) once nlist and the
        training sample both get large, otherwise full-batch training time
        grows with nlist * train_n and becomes the actual bottleneck long
        before insertion does. PQ's own per-subspace kemeans is unaffected,
        its k=256 is small and constant regardless of dataset size.

        pq_train_size optionally caps the representative sample used for PQ
        codebooks.  PQ has only 256 centroids per subspace, so it does not
        need the much larger sample used to seed tens of thousands of IVF
        centroids.  The benchmark exposes this separately so recall can be
        checked before adopting the faster setting in production."""
        training_vectors = np.asarray(training_vectors, dtype=np.float32)
        assert len(training_vectors) >= self.nlist, \
            f"need at least nlist={self.nlist} training vectors, got {len(training_vectors)}"

        if coarse_training not in {"legacy", "accumulated"}:
            raise ValueError("coarse_training must be 'legacy' or 'accumulated'")
        self.coarse_training = coarse_training
        self.centroids = _kmeans(
            training_vectors, self.nlist, n_iters, self.rng,
            minibatch_size=minibatch_size, tol=tol,
            training_mode=coarse_training)
        self._centroid_norms = (self.centroids ** 2).sum(axis=1)
        pq_training_vectors = training_vectors
        if pq_train_size is not None and pq_train_size < len(training_vectors):
            pq_idx = self.rng.choice(
                len(training_vectors), size=pq_train_size, replace=False)
            pq_training_vectors = training_vectors[pq_idx]
        if self.pq_mode == "residual":
            assignments = self._nearest_clusters(pq_training_vectors)
            pq_training_vectors = (
                pq_training_vectors - self.centroids[assignments])
        self.pq.train(pq_training_vectors, n_iters=n_iters)
        self._trained = True
        self._close_packed()
        self._close_raw()
        self._packed_offsets = None
        self._cluster_sizes.fill(0)
        self._base_cluster_sizes.fill(0)
        self._raw_count = 0
        self._mlx_router = None

        for path in (self._packed_path(), self._packed_path() + ".tmp"):
            if os.path.exists(path):
                os.remove(path)
        if os.path.exists(self._raw_path()):
            os.remove(self._raw_path())
        if self.store_full_vectors:
            open(self._raw_path(), "wb").close()

        # start every cluster's file empty
        for c in range(self.nlist):
            open(self._cluster_path(c), "wb").close()
            delta_path = self._delta_path(c)
            if os.path.exists(delta_path):
                os.remove(delta_path)

    def _nearest_cluster(self, vector):
        return int(self._nearest_clusters(vector[None, :], batch_size=1)[0])

    def _nearest_clusters(self, vectors, batch_size=None):
        """Route a batch with bounded-memory matrix multiplications.

        The scalar implementation performed one NumPy call per vector.  This
        version does the same exact L2 assignment while letting BLAS process a
        block at once.  ``batch_size`` is chosen to cap the distance matrix at
        roughly 64 MiB, so it remains safe when ``nlist`` is large.
        """
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with shape (n, {self.dim}), got {vectors.shape}")
        if self.routing_backend == "mlx":
            return self._nearest_clusters_mlx(vectors, batch_size)

        if batch_size is None:
            target_bytes = 64 * 1024 * 1024
            batch_size = max(1, min(8192, target_bytes // (4 * self.nlist)))
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        centroid_sq = self._centroid_norms
        assignments = np.empty(len(vectors), dtype=np.int64)
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            chunk = vectors[start:end]
            dists = chunk @ self.centroids.T
            dists *= -2.0
            dists += (chunk ** 2).sum(axis=1)[:, None]
            dists += centroid_sq[None, :]
            assignments[start:end] = dists.argmin(axis=1)
        return assignments

    def _nearest_clusters_mlx(self, vectors, batch_size=None):
        """Route ingestion batches on Apple Silicon's Metal GPU via MLX.

        Only the coarse IVF assignment is offloaded. PQ encoding and posting
        writes remain on the CPU, so the on-disk format and search behavior
        are identical to the NumPy backend.
        """
        if batch_size is None:
            batch_size = 1024
        if self._mlx_router is None:
            from .mlx_routing import MLXRouter
            self._mlx_router = MLXRouter(self.centroids)
        return self._mlx_router.route(vectors, batch_size=batch_size)

    def add(self, vector_id: int, vector: np.ndarray):
        """Assign to nearest cluster, PQ-encode, APPEND straight to that
        cluster's file on disk. The full dataset is never accumulated in
        memory, not even transiently, each vector is written and forgotten."""
        assert self._trained, "call train() first"
        self.add_batch(np.asarray(vector, dtype=np.float32)[None, :],
                       np.asarray([vector_id], dtype=np.int64))

    def add_batch(self, vectors: np.ndarray, ids: np.ndarray,
                  assignment_batch_size: int = None,
                  pq_batch_size: int = 20000):
        """Route, encode, and append many vectors as one ingestion unit.

        This removes the three dominant per-vector costs in ``add``: a scalar
        centroid scan, a scalar PQ encode, and an open/write/close cycle.  The
        output format is unchanged, so existing indexes remain readable and
        scalar ``add`` remains available as a compatibility wrapper.
        """
        assert self._trained, "call train() first"
        vectors = np.asarray(vectors, dtype=np.float32)
        ids = np.asarray(ids, dtype=np.int64)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"expected vectors with shape (n, {self.dim}), got {vectors.shape}")
        if ids.ndim != 1 or len(ids) != len(vectors):
            raise ValueError("ids must be one-dimensional and match the number of vectors")
        if len(vectors) == 0:
            return

        cluster_ids = self._nearest_clusters(vectors, assignment_batch_size)
        pq_vectors = vectors
        if self.pq_mode == "residual":
            pq_vectors = vectors - self.centroids[cluster_ids]
        codes = self.pq.encode(pq_vectors, batch_size=pq_batch_size)

        raw_rows = None
        if self.store_full_vectors:
            self._close_raw()
            raw_rows = np.arange(
                self._raw_count, self._raw_count + len(vectors), dtype=np.int64)
            with open(self._raw_path(), "ab") as raw_file:
                raw_file.write(np.ascontiguousarray(vectors, dtype="<f4").tobytes())
            self._raw_count += len(vectors)

        records = np.empty(len(vectors), dtype=self._record_dtype)
        records["id"] = ids
        records["code"] = codes
        if self.store_full_vectors:
            records["raw_row"] = raw_rows

        # Sorting once turns every cluster's records into a contiguous slice.
        # Each non-empty cluster is opened once per user batch, rather than
        # opening a file once per vector.
        order = np.argsort(cluster_ids, kind="stable")
        sorted_clusters = cluster_ids[order]
        sorted_records = records[order]
        unique_clusters, starts, counts = np.unique(
            sorted_clusters, return_index=True, return_counts=True)
        for cluster_id, start, count in zip(unique_clusters, starts, counts):
            cluster_id = int(cluster_id)
            path = (self._delta_path(cluster_id) if self.is_compacted
                    else self._cluster_path(cluster_id))
            with open(path, "ab") as f:
                f.write(sorted_records[start:start + count].tobytes())

        self._cluster_sizes += np.bincount(cluster_ids, minlength=self.nlist)

    def _empty_records(self):
        return np.empty(0, dtype=self._record_dtype)

    def _read_records_file(self, path):
        with open(path, "rb") as f:
            raw = f.read()
        if len(raw) == 0:
            return self._empty_records()
        if len(raw) % self._record_dtype.itemsize != 0:
            raise ValueError(f"corrupt posting list {path}: partial record")
        return np.frombuffer(raw, dtype=self._record_dtype)

    def _read_cluster(self, cluster_id):
        """Return one posting list from the packed base plus mutable delta.

        Before compact(), this reads the legacy per-cluster file.  After
        compact(), the immutable base is a zero-open memory-map slice; a file
        is opened only when that cluster has received new delta records since
        the last compaction.
        """
        if not self.is_compacted:
            return self._read_records_file(self._cluster_path(cluster_id))

        start = int(self._packed_offsets[cluster_id])
        end = int(self._packed_offsets[cluster_id + 1])
        base = self._packed_records[start:end]

        delta_count = int(
            self._cluster_sizes[cluster_id] - self._base_cluster_sizes[cluster_id])
        if delta_count == 0:
            return base

        delta = self._read_records_file(self._delta_path(cluster_id))
        if len(delta) != delta_count:
            raise ValueError(
                f"delta size mismatch for cluster {cluster_id}: expected "
                f"{delta_count} records, found {len(delta)}")
        if len(base) == 0:
            return delta
        return np.concatenate((base, delta))

    def compact(self):
        """Pack all posting lists into one memory-mapped immutable segment.

        Search then slices ``postings.bin`` using an in-memory offset table,
        eliminating one open/read/close cycle per probed cluster.  Inserts made
        later go to small delta files; calling compact() again merges them.
        """
        assert self._trained, "call train() first"
        packed_path = self._packed_path()
        temp_path = packed_path + ".tmp"
        offsets = np.zeros(self.nlist + 1, dtype=np.int64)
        cursor = 0

        with open(temp_path, "wb") as output:
            for cluster_id in range(self.nlist):
                offsets[cluster_id] = cursor
                expected_count = int(self._cluster_sizes[cluster_id])

                if self.is_compacted:
                    start = int(self._packed_offsets[cluster_id])
                    end = int(self._packed_offsets[cluster_id + 1])
                    base = self._packed_records[start:end]
                    if len(base):
                        output.write(base.tobytes())
                    delta_count = expected_count - len(base)
                    if delta_count:
                        delta_path = self._delta_path(cluster_id)
                        actual_delta_bytes = os.path.getsize(delta_path)
                        expected_delta_bytes = delta_count * self._record_dtype.itemsize
                        if actual_delta_bytes != expected_delta_bytes:
                            raise ValueError(
                                f"delta size mismatch for cluster {cluster_id}")
                        with open(delta_path, "rb") as source:
                            shutil.copyfileobj(source, output)
                else:
                    source_path = self._cluster_path(cluster_id)
                    actual_bytes = os.path.getsize(source_path)
                    expected_bytes = expected_count * self._record_dtype.itemsize
                    if actual_bytes != expected_bytes:
                        raise ValueError(
                            f"posting size mismatch for cluster {cluster_id}: "
                            f"expected {expected_bytes}, found {actual_bytes}")
                    if actual_bytes:
                        with open(source_path, "rb") as source:
                            shutil.copyfileobj(source, output)

                cursor += expected_count

            offsets[-1] = cursor
            output.flush()
            os.fsync(output.fileno())

        self._close_packed()
        os.replace(temp_path, packed_path)
        self._packed_offsets = offsets
        self._base_cluster_sizes = self._cluster_sizes.copy()
        self._map_packed()

        # Commit metadata before deleting the now-redundant source files.  If
        # the process stops earlier, the old layout is still recoverable.
        self.save()
        for cluster_id in range(self.nlist):
            for path in (self._cluster_path(cluster_id), self._delta_path(cluster_id)):
                if os.path.exists(path):
                    os.remove(path)

        return {
            "vectors": cursor,
            "bytes": os.path.getsize(packed_path),
            "path": packed_path,
        }

    def _select_probe_clusters(self, query, nprobe=None, max_candidates=None):
        """Select posting lists by count or by their accumulated record size.

        Candidate-budget routing makes configurations with different ``nlist``
        directly comparable: it follows centroid-distance order and stops once
        the selected lists contain at least the requested number of records.
        The last list may take the actual count slightly over the budget.
        """
        if nprobe is not None and max_candidates is not None:
            raise ValueError("nprobe and max_candidates are mutually exclusive")
        if nprobe is None and max_candidates is None:
            nprobe = 8
        if nprobe is not None and nprobe <= 0:
            raise ValueError("nprobe must be positive")
        if max_candidates is not None and max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

        cluster_dists = self._centroid_norms.copy()
        cluster_dists -= 2.0 * (self.centroids @ query)
        if max_candidates is not None:
            order = np.argsort(cluster_dists)
            cumulative = np.cumsum(self._cluster_sizes[order], dtype=np.int64)
            stop = int(np.searchsorted(cumulative, max_candidates, side="left")) + 1
            stop = min(max(stop, 1), self.nlist)
            probe_clusters = order[:stop]
        else:
            nprobe = min(int(nprobe), self.nlist)
            if nprobe == self.nlist:
                probe_clusters = np.argsort(cluster_dists)
            else:
                probe_clusters = np.argpartition(cluster_dists, nprobe - 1)[:nprobe]
                probe_clusters = probe_clusters[np.argsort(cluster_dists[probe_clusters])]
        scanned = int(self._cluster_sizes[probe_clusters].sum())
        return probe_clusters, scanned

    def _approximate_shortlist(self, query, shortlist_size, nprobe=None,
                               max_candidates=None, scan_chunk_size=65536):
        """Stream selected postings and retain only the best global shortlist."""
        if shortlist_size <= 0:
            raise ValueError("shortlist_size must be positive")
        if scan_chunk_size <= 0:
            raise ValueError("scan_chunk_size must be positive")
        probe_clusters, scanned = self._select_probe_clusters(
            query, nprobe=nprobe, max_candidates=max_candidates)
        best_records = self._empty_records()
        best_dists = np.empty(0, dtype=np.float32)
        standard_table = (
            self.pq.distance_table(query) if self.pq_mode == "standard" else None)

        for cluster_id in probe_clusters:
            cluster_id = int(cluster_id)
            records = self._read_cluster(cluster_id)
            if len(records) == 0:
                continue
            table = standard_table
            if self.pq_mode == "residual":
                table = self.pq.distance_table(query - self.centroids[cluster_id])
            for start in range(0, len(records), scan_chunk_size):
                chunk = records[start:start + scan_chunk_size]
                dists = self.pq.asymmetric_distances(chunk["code"], table)
                local_size = min(shortlist_size, len(chunk))
                local = np.argpartition(dists, local_size - 1)[:local_size]
                candidate_records = np.concatenate((best_records, chunk[local]))
                candidate_dists = np.concatenate((best_dists, dists[local]))
                keep_size = min(shortlist_size, len(candidate_dists))
                keep = np.argpartition(candidate_dists, keep_size - 1)[:keep_size]
                best_records = candidate_records[keep]
                best_dists = candidate_dists[keep]

        if len(best_dists):
            order = np.argsort(best_dists)
            best_records = best_records[order]
            best_dists = best_dists[order]
        return best_records, best_dists, scanned

    def search(self, query, k=10, nprobe=None, max_candidates=None, rerank=0):
        """Search using either a probe count or an approximate scan budget.

        If neither routing control is supplied, eight posting lists are
        probed for backward compatibility. Supplying both is an error.
        ``rerank`` selects that many PQ finalists for exact raw-vector L2.
        """
        assert self._trained, "call train() first"
        query = np.asarray(query, dtype=np.float32)

        if query.shape != (self.dim,):
            raise ValueError(f"expected a flat query of dim {self.dim}, got {query.shape}")
        if k <= 0 or rerank < 0:
            raise ValueError("k must be positive and rerank must be non-negative")
        if rerank and not self.store_full_vectors:
            raise ValueError(
                "reranking requires an index created with store_full_vectors=True")
        shortlist_size = max(k, rerank) if rerank else k
        records, approximate_dists, _ = self._approximate_shortlist(
            query, shortlist_size, nprobe=nprobe,
            max_candidates=max_candidates)
        if len(records) == 0:
            return [], []
        k = min(k, len(records))

        if rerank:
            self._map_raw()
            raw_rows = records["raw_row"]
            exact_vectors = np.asarray(self._raw_vectors[raw_rows])
            exact_dists = ((exact_vectors - query) ** 2).sum(axis=1)
            final = np.argpartition(exact_dists, k - 1)[:k]
            final = final[np.argsort(exact_dists[final])]
            return records["id"][final].tolist(), exact_dists[final].tolist()

        return records["id"][:k].tolist(), approximate_dists[:k].tolist()

    def resident_memory_bytes(self):
        """What's ACTUALLY kept in RAM, independent of how many vectors
        have been added. This should not grow with n, only with nlist."""
        packed_metadata_bytes = self._base_cluster_sizes.nbytes
        if self._packed_offsets is not None:
            packed_metadata_bytes += self._packed_offsets.nbytes
        mlx_cache_bytes = (
            self._mlx_router.cache_bytes if self._mlx_router is not None else 0)
        return (self.centroids.nbytes + self._centroid_norms.nbytes +
                self.pq.codebooks.nbytes + self._cluster_sizes.nbytes +
                packed_metadata_bytes + mlx_cache_bytes)

    def total_vectors(self):
        return int(self._cluster_sizes.sum())

    def close(self):
        """Release the persistent memory map, if this index is compacted."""
        self._close_packed()
        self._close_raw()

    def posting_bytes(self):
        """Bytes used by posting records, excluding raw vectors and metadata."""
        if self.is_compacted:
            total = os.path.getsize(self._packed_path())
            for cluster_id in np.flatnonzero(
                    self._cluster_sizes > self._base_cluster_sizes):
                total += os.path.getsize(self._delta_path(int(cluster_id)))
            return total
        return sum(
            os.path.getsize(self._cluster_path(cluster_id))
            for cluster_id in range(self.nlist))

    def raw_vector_bytes(self):
        if not self.store_full_vectors:
            return 0
        return os.path.getsize(self._raw_path()) if os.path.exists(self._raw_path()) else 0

    def disk_bytes(self):
        return self.posting_bytes() + self.raw_vector_bytes()

    def save(self, path: str = None):
        """Persist the routing index, centroids, PQ codebooks, and cluster
        sizes, to disk. This is the piece that was actually missing for
        this to be a real database instead of an in-memory structure that
        happens to write vectors to disk: the posting-list .bin files
        already ARE persistent (they're just files, they survive a
        process exit on their own), but without this, nothing records
        which cluster is which, or how to PQ-encode a future query the
        same way training did. The data would just be orphaned bytes on
        disk with no way back in.

        Defaults to a file inside storage_dir, so the whole index (routing
        index + posting lists) lives together in one self-contained folder."""
        assert self._trained, "nothing to save, call train() first"
        path = path or os.path.join(self.storage_dir, "index_meta.npz")
        payload = dict(
            storage_version=3,
            centroids=self.centroids,
            codebooks=self.pq.codebooks,
            cluster_sizes=self._cluster_sizes,
            dim=self.dim,
            nlist=self.nlist,
            pq_m=self.pq.m,
            pq_k=self.pq.k,
            store_full_vectors=self.store_full_vectors,
            raw_count=self._raw_count,
            routing_backend=self.routing_backend,
            pq_mode=self.pq_mode,
            coarse_training=self.coarse_training,
        )
        if self.is_compacted:
            payload["packed_offsets"] = self._packed_offsets
            payload["base_cluster_sizes"] = self._base_cluster_sizes
        np.savez(path, **payload)
        return path

    @classmethod
    def load(cls, storage_dir: str, path: str = None,
             routing_backend: str = None):
        """Reconstruct an index from a previous save(), pointed at the
        SAME storage_dir its posting-list files already live in. Note
        this does NOT call train(), train() is what truncates every
        cluster file to start fresh, calling it here would wipe out the
        exact data this method exists to recover. Loading only ever
        reads: it re-creates the centroids and codebooks in memory and
        leaves every .bin file on disk untouched, so add() can keep
        appending to them right where the last process left off."""
        path = path or os.path.join(storage_dir, "index_meta.npz")
        data = np.load(path)
        store_full_vectors = (
            bool(data["store_full_vectors"])
            if "store_full_vectors" in data.files else False)
        if routing_backend is None:
            routing_backend = (
                str(data["routing_backend"])
                if "routing_backend" in data.files else "numpy")
        pq_mode = str(data["pq_mode"]) if "pq_mode" in data.files else "standard"
        idx = cls(int(data["dim"]), int(data["nlist"]),
                  pq_m=int(data["pq_m"]), pq_k=int(data["pq_k"]),
                  storage_dir=storage_dir,
                  store_full_vectors=store_full_vectors,
                  routing_backend=routing_backend,
                  pq_mode=pq_mode)
        idx.coarse_training = (
            str(data["coarse_training"])
            if "coarse_training" in data.files else "legacy")
        idx.centroids = data["centroids"]
        idx._centroid_norms = (idx.centroids ** 2).sum(axis=1)
        idx.pq.codebooks = data["codebooks"]
        idx._cluster_sizes = data["cluster_sizes"]
        idx._raw_count = int(data["raw_count"]) if "raw_count" in data.files else 0
        if "packed_offsets" in data.files:
            idx._packed_offsets = data["packed_offsets"].astype(np.int64, copy=True)
            idx._base_cluster_sizes = data["base_cluster_sizes"].astype(
                np.int64, copy=True)
            idx._map_packed()
        idx._trained = True
        return idx
