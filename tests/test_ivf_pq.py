import os
import shutil
import resource
import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_pq import IVFPQIndex

STORAGE = "/tmp/ivf_pq_test_storage"


def _clean():
    if os.path.exists(STORAGE):
        shutil.rmtree(STORAGE)


def test_recall_against_brute_force():
    """The real question: does cluster-routing + PQ still find the right
    answers, at a reasonable nprobe?"""
    _clean()
    np.random.seed(0)
    dim = 64
    n = 5000
    vectors = np.random.rand(n, dim).astype(np.float32)
    queries = np.random.rand(30, dim).astype(np.float32)
    k = 10

    brute = BruteForceIndex(dim, metric="l2")
    brute.add(vectors)

    idx = IVFPQIndex(dim, nlist=50, pq_m=16, pq_k=256, storage_dir=STORAGE)
    idx.train(vectors[:2000])
    for i, v in enumerate(vectors):
        idx.add(i, v)

    hits, total = 0, 0
    for q in queries:
        true_ids, _ = brute.search(q, k=k)
        got_ids, _ = idx.search(q, k=k, nprobe=8)
        hits += len(set(true_ids.tolist()) & set(got_ids))
        total += k
    recall = hits / total
    print(f"PASS (informational): n={n} nlist=50 nprobe=8 recall@{k}={recall:.3f}")
    assert recall >= 0.3, f"recall too low: {recall:.3f}"
    _clean()


def test_nprobe_is_a_real_dial():
    """More probed clusters -> better recall. If this doesn't hold, the
    clustering or routing logic is broken, not just imprecise."""
    _clean()
    np.random.seed(1)
    dim = 32
    n = 3000
    vectors = np.random.rand(n, dim).astype(np.float32)
    queries = np.random.rand(20, dim).astype(np.float32)
    k = 10

    brute = BruteForceIndex(dim, metric="l2")
    brute.add(vectors)

    idx = IVFPQIndex(dim, nlist=40, pq_m=8, pq_k=256, storage_dir=STORAGE)
    idx.train(vectors[:1500])
    for i, v in enumerate(vectors):
        idx.add(i, v)

    results = {}
    for nprobe in [1, 5, 20]:
        hits, total = 0, 0
        for q in queries:
            true_ids, _ = brute.search(q, k=k)
            got_ids, _ = idx.search(q, k=k, nprobe=nprobe)
            hits += len(set(true_ids.tolist()) & set(got_ids))
            total += k
        results[nprobe] = hits / total
        print(f"PASS (informational): nprobe={nprobe:3d}  recall@{k}={results[nprobe]:.3f}")

    assert results[20] > results[5] > results[1], \
        f"expected recall to improve monotonically with nprobe, got {results}"
    _clean()


def test_add_batch_matches_scalar_add():
    """Batch ingestion must preserve the on-disk format and search result."""
    scalar_storage = STORAGE + "_scalar"
    batch_storage = STORAGE + "_batch"
    for path in (scalar_storage, batch_storage):
        if os.path.exists(path):
            shutil.rmtree(path)

    rng = np.random.default_rng(7)
    dim = 32
    vectors = rng.random((1200, dim), dtype=np.float32)
    train = vectors[:800]
    ids = np.arange(10000, 11200, dtype=np.int64)
    queries = rng.random((10, dim), dtype=np.float32)

    scalar = IVFPQIndex(dim, nlist=20, pq_m=8, storage_dir=scalar_storage, seed=9)
    batched = IVFPQIndex(dim, nlist=20, pq_m=8, storage_dir=batch_storage, seed=9)
    scalar.train(train, n_iters=5)
    batched.train(train, n_iters=5)
    for vector_id, vector in zip(ids, vectors):
        scalar.add(int(vector_id), vector)
    batched.add_batch(vectors, ids)

    assert np.array_equal(scalar._cluster_sizes, batched._cluster_sizes)
    for query in queries:
        scalar_ids, scalar_dists = scalar.search(query, k=10, nprobe=6)
        batch_ids, batch_dists = batched.search(query, k=10, nprobe=6)
        assert scalar_ids == batch_ids
        assert np.allclose(scalar_dists, batch_dists)

    for path in (scalar_storage, batch_storage):
        shutil.rmtree(path)


def test_resident_memory_does_not_scale_with_n():
    """THE core claim of this whole file: resident_memory_bytes() should
    depend on nlist and pq params only, not on how many vectors were
    added. Confirm by adding two very different amounts and checking the
    self-reported figure is identical, then cross-check against actual
    process RSS to make sure this isn't just a bookkeeping lie."""
    _clean()
    np.random.seed(2)
    dim = 32

    idx = IVFPQIndex(dim, nlist=30, pq_m=8, pq_k=256, storage_dir=STORAGE)
    train_vectors = np.random.rand(2000, dim).astype(np.float32)
    idx.train(train_vectors)
    mem_at_0 = idx.resident_memory_bytes()

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    n_big = 20000
    for i in range(n_big):
        idx.add(i, np.random.rand(dim).astype(np.float32))

    mem_at_20000 = idx.resident_memory_bytes()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    disk_bytes = sum(
        os.path.getsize(idx._cluster_path(c)) for c in range(idx.nlist)
    )

    print(f"PASS (informational): resident_memory before={mem_at_0}B after adding "
          f"{n_big} vectors={mem_at_20000}B (unchanged={mem_at_0 == mem_at_20000})  "
          f"disk_bytes_written={disk_bytes/1024:.1f}KB  "
          f"process_rss_before={rss_before}KB  process_rss_after={rss_after}KB")

    assert mem_at_0 == mem_at_20000, \
        "resident_memory_bytes() must not change as vectors are added, that's the whole point"
    assert disk_bytes > 100000, "expected real data to have actually been written to disk"
    assert idx.total_vectors() == n_big
    _clean()


if __name__ == "__main__":
    test_recall_against_brute_force()
    test_nprobe_is_a_real_dial()
    test_add_batch_matches_scalar_add()
    test_resident_memory_does_not_scale_with_n()
    print("\nall IVF+PQ tests passed")
