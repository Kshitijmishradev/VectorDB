import os
import shutil

import numpy as np

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_pq import IVFPQIndex


STORAGE = "/tmp/ivf_pq_rerank_test_storage"


def _clean(index=None):
    if index is not None:
        index.close()
    if os.path.exists(STORAGE):
        shutil.rmtree(STORAGE)


def test_full_shortlist_rerank_matches_exact_search():
    _clean()
    rng = np.random.default_rng(31)
    dim = 32
    n = 2500
    vectors = rng.random((n, dim), dtype=np.float32)
    queries = rng.random((12, dim), dtype=np.float32)

    brute = BruteForceIndex(dim, metric="l2")
    brute.add(vectors)
    index = IVFPQIndex(
        dim, nlist=30, pq_m=8, storage_dir=STORAGE, seed=7,
        store_full_vectors=True)
    index.train(vectors[:1500], n_iters=5)
    index.add_batch(vectors, np.arange(n, dtype=np.int64))
    index.compact()

    assert index.raw_vector_bytes() == n * dim * 4
    assert index.posting_bytes() == n * (8 + index.pq.m + 8)

    for query in queries:
        expected_ids, expected_dists = brute.search(query, k=10)
        actual_ids, actual_dists = index.search(
            query, k=10, nprobe=index.nlist, rerank=n)
        assert actual_ids == expected_ids.tolist()
        assert np.allclose(actual_dists, expected_dists)

    print("PASS: full-shortlist reranking matches exact brute-force search")
    _clean(index)


def test_rerank_requires_raw_vectors():
    _clean()
    rng = np.random.default_rng(37)
    vectors = rng.random((500, 16), dtype=np.float32)
    index = IVFPQIndex(16, nlist=10, pq_m=4, storage_dir=STORAGE)
    index.train(vectors[:300], n_iters=3)
    index.add_batch(vectors, np.arange(len(vectors), dtype=np.int64))
    try:
        index.search(vectors[0], k=10, nprobe=5, rerank=50)
    except ValueError as exc:
        assert "store_full_vectors" in str(exc)
    else:
        raise AssertionError("reranking without raw vectors should fail")

    print("PASS: reranking rejects compressed-only indexes")
    _clean(index)


def test_raw_vectors_survive_load_delta_add_and_recompact():
    _clean()
    rng = np.random.default_rng(41)
    first = rng.random((1000, 16), dtype=np.float32)
    second = rng.random((100, 16), dtype=np.float32)

    index = IVFPQIndex(
        16, nlist=16, pq_m=4, storage_dir=STORAGE,
        store_full_vectors=True)
    index.train(first[:600], n_iters=3)
    index.add_batch(first, np.arange(len(first), dtype=np.int64))
    index.compact()
    index.close()

    loaded = IVFPQIndex.load(STORAGE)
    second_ids = np.arange(len(first), len(first) + len(second), dtype=np.int64)
    loaded.add_batch(second, second_ids)
    loaded.save()
    found, distances = loaded.search(
        second[0], k=1, nprobe=loaded.nlist, rerank=100)
    assert found == [int(second_ids[0])]
    assert np.isclose(distances[0], 0.0)

    loaded.compact()
    loaded.close()
    reloaded = IVFPQIndex.load(STORAGE)
    found, distances = reloaded.search(
        second[0], k=1, nprobe=reloaded.nlist, rerank=100)
    assert found == [int(second_ids[0])]
    assert np.isclose(distances[0], 0.0)
    assert reloaded.raw_vector_bytes() == (len(first) + len(second)) * 16 * 4

    print("PASS: raw-vector persistence, delta insertion, and re-compaction work")
    _clean(reloaded)


if __name__ == "__main__":
    test_full_shortlist_rerank_matches_exact_search()
    test_rerank_requires_raw_vectors()
    test_raw_vectors_survive_load_delta_add_and_recompact()
    print("\nall IVF+PQ reranking tests passed")
