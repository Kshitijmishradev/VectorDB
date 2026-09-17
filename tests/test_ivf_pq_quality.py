import os
import tempfile

import numpy as np

from vectordb.ivf_pq import IVFPQIndex


def _data(seed=41, n=800, dim=16):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(8, dim)).astype(np.float32)
    labels = rng.integers(0, len(centers), size=n)
    vectors = centers[labels] + 0.15 * rng.normal(size=(n, dim)).astype(np.float32)
    return vectors.astype(np.float32), rng


def _build(storage, pq_mode="standard", raw=False):
    vectors, rng = _data()
    index = IVFPQIndex(
        16, nlist=16, pq_m=4, pq_k=16, storage_dir=storage,
        seed=7, store_full_vectors=raw, pq_mode=pq_mode)
    index.train(vectors[:500], n_iters=5, pq_train_size=400)
    index.add_batch(vectors, np.arange(len(vectors), dtype=np.int64))
    return index, vectors, rng


def test_residual_codes_are_cluster_relative():
    with tempfile.TemporaryDirectory() as storage:
        index, vectors, _ = _build(storage, pq_mode="residual")
        assignments = index._nearest_clusters(vectors)
        expected = index.pq.encode(vectors - index.centroids[assignments])
        actual = np.empty_like(expected)
        for cluster_id in range(index.nlist):
            rows = np.flatnonzero(assignments == cluster_id)
            records = index._read_cluster(cluster_id)
            if len(rows):
                actual[rows] = records["code"]
        assert np.array_equal(actual, expected)


def test_candidate_budget_routing_and_validation():
    with tempfile.TemporaryDirectory() as storage:
        index, vectors, _ = _build(storage)
        query = vectors[0]
        clusters, scanned = index._select_probe_clusters(
            query, max_candidates=125)
        assert scanned >= 125
        if len(clusters) > 1:
            assert scanned - index._cluster_sizes[clusters[-1]] < 125
        ids, _ = index.search(query, k=10, max_candidates=125)
        assert len(ids) == 10
        try:
            index.search(query, k=10, nprobe=2, max_candidates=125)
        except ValueError as exc:
            assert "mutually exclusive" in str(exc)
        else:
            raise AssertionError("supplying both routing controls must fail")


def test_streaming_top_n_matches_full_concatenation():
    with tempfile.TemporaryDirectory() as storage:
        index, vectors, rng = _build(storage)
        query = rng.normal(size=16).astype(np.float32)
        clusters, _ = index._select_probe_clusters(query, nprobe=7)
        records = np.concatenate([index._read_cluster(int(c)) for c in clusters])
        table = index.pq.distance_table(query)
        distances = index.pq.asymmetric_distances(records["code"], table)
        expected = np.argsort(distances)[:25]
        streamed, streamed_distances, _ = index._approximate_shortlist(
            query, 25, nprobe=7, scan_chunk_size=11)
        assert np.allclose(streamed_distances, distances[expected])
        # PQ produces many equal distances. Different tied IDs are valid, but
        # every retained item must be no farther than the full-scan cutoff.
        assert np.all(streamed_distances <= distances[expected][-1])


def test_residual_reranking_matches_exact_with_all_lists():
    with tempfile.TemporaryDirectory() as storage:
        index, vectors, rng = _build(storage, pq_mode="residual", raw=True)
        query = rng.normal(size=16).astype(np.float32)
        truth = np.argsort(((vectors - query) ** 2).sum(axis=1))[:10]
        ids, distances = index.search(
            query, k=10, nprobe=index.nlist, rerank=len(vectors))
        assert ids == truth.tolist()
        assert np.allclose(distances, ((vectors[truth] - query) ** 2).sum(axis=1))


def test_metadata_v3_and_legacy_defaults():
    with tempfile.TemporaryDirectory() as storage:
        index, _, _ = _build(storage, pq_mode="residual")
        metadata = index.save()
        with np.load(metadata) as saved:
            assert int(saved["storage_version"]) == 3
            assert str(saved["pq_mode"]) == "residual"

        # Model a v2 metadata file by removing the new fields. The posting
        # layout is unchanged, so it must still load as standard/legacy.
        legacy_storage = os.path.join(storage, "legacy")
        os.makedirs(legacy_storage)
        standard = IVFPQIndex(
            16, nlist=16, pq_m=4, pq_k=16,
            storage_dir=legacy_storage, seed=7)
        vectors, _ = _data()
        standard.train(vectors[:500], n_iters=3)
        standard.add_batch(vectors[:100], np.arange(100))
        standard.save()
        legacy_path = os.path.join(legacy_storage, "index_meta.npz")
        with np.load(legacy_path) as saved:
            payload = {
                key: saved[key] for key in saved.files
                if key not in {"storage_version", "pq_mode", "coarse_training"}
            }
        np.savez(legacy_path, **payload)
        loaded = IVFPQIndex.load(legacy_storage)
        assert loaded.pq_mode == "standard"
        assert loaded.coarse_training == "legacy"
        assert loaded.total_vectors() == 100


def test_accumulated_training_and_residual_delta_compaction():
    with tempfile.TemporaryDirectory() as storage:
        vectors, rng = _data(n=900)
        index = IVFPQIndex(
            16, nlist=16, pq_m=4, pq_k=16, storage_dir=storage,
            seed=9, store_full_vectors=True, pq_mode="residual")
        index.train(
            vectors[:500], n_iters=5, minibatch_size=150,
            coarse_training="accumulated")
        assert index.coarse_training == "accumulated"
        assert np.isfinite(index.centroids).all()
        index.add_batch(vectors[:800], np.arange(800))
        index.compact()
        index.add_batch(vectors[800:], np.arange(800, 900))
        index.save()
        query = vectors[850]
        before = index.search(query, k=10, nprobe=index.nlist, rerank=50)
        index.compact()
        index.close()
        loaded = IVFPQIndex.load(storage)
        after = loaded.search(query, k=10, nprobe=loaded.nlist, rerank=50)
        assert before[0] == after[0]
        assert np.allclose(before[1], after[1])
        assert loaded.pq_mode == "residual"
        assert loaded.coarse_training == "accumulated"
