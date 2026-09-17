import os
import shutil
import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_pq import IVFPQIndex

STORAGE = "/tmp/ivf_pq_persist_test_storage"


def _clean():
    if os.path.exists(STORAGE):
        shutil.rmtree(STORAGE)


def test_save_then_load_gives_identical_search_results():
    """The real test of persistence: build an index, save it, throw away
    the in-memory object entirely (simulating a process restart), load a
    brand new object from disk, and confirm it answers queries exactly
    the same as the original. If this doesn't match, persistence is
    lying about what it saved."""
    _clean()
    np.random.seed(0)
    dim = 32
    n = 2000
    vectors = np.random.rand(n, dim).astype(np.float32)
    queries = np.random.rand(10, dim).astype(np.float32)

    idx = IVFPQIndex(dim, nlist=20, pq_m=8, pq_k=256, storage_dir=STORAGE)
    idx.train(vectors[:1000])
    for i, v in enumerate(vectors):
        idx.add(i, v)

    # capture results BEFORE save/reload
    before = [idx.search(q, k=5, nprobe=5) for q in queries]

    saved_path = idx.save()
    assert os.path.exists(saved_path)

    del idx  # simulate the process actually exiting

    loaded = IVFPQIndex.load(STORAGE)
    after = [loaded.search(q, k=5, nprobe=5) for q in queries]

    for (ids_b, dists_b), (ids_a, dists_a) in zip(before, after):
        assert ids_b == ids_a, f"ids changed after reload: {ids_b} vs {ids_a}"
        assert np.allclose(dists_b, dists_a), f"distances changed after reload: {dists_b} vs {dists_a}"

    print("PASS: search results identical before save and after load")
    _clean()


def test_loaded_index_reports_correct_total_vectors_and_memory():
    """Loading shouldn't just answer queries right, it should also report
    accurate bookkeeping (total_vectors, resident_memory_bytes), since a
    real caller would check these before deciding to add more data."""
    _clean()
    np.random.seed(1)
    dim = 16
    n = 800
    vectors = np.random.rand(n, dim).astype(np.float32)

    idx = IVFPQIndex(dim, nlist=10, pq_m=4, pq_k=256, storage_dir=STORAGE)
    idx.train(vectors[:400])
    for i, v in enumerate(vectors):
        idx.add(i, v)
    idx.save()
    mem_before = idx.resident_memory_bytes()
    total_before = idx.total_vectors()
    del idx

    loaded = IVFPQIndex.load(STORAGE)
    assert loaded.total_vectors() == total_before == n
    assert loaded.resident_memory_bytes() == mem_before
    print(f"PASS: total_vectors={loaded.total_vectors()} resident_memory_bytes={loaded.resident_memory_bytes()} "
          f"both match pre-save values")
    _clean()


def test_can_keep_adding_after_load():
    """A persisted index isn't read-only, you should be able to reopen it
    and keep inserting, with the new vectors landing in the SAME
    posting-list files the old ones are in, not a fresh empty index."""
    _clean()
    np.random.seed(2)
    dim = 16
    n_first = 500
    n_second = 500
    vectors1 = np.random.rand(n_first, dim).astype(np.float32)
    vectors2 = np.random.rand(n_second, dim).astype(np.float32)

    idx = IVFPQIndex(dim, nlist=10, pq_m=4, pq_k=256, storage_dir=STORAGE)
    idx.train(vectors1[:300])
    for i, v in enumerate(vectors1):
        idx.add(i, v)
    idx.save()
    del idx

    loaded = IVFPQIndex.load(STORAGE)
    assert loaded.total_vectors() == n_first
    for i, v in enumerate(vectors2):
        loaded.add(n_first + i, v)  # new, non-colliding ids

    assert loaded.total_vectors() == n_first + n_second

    # sanity: a vector added AFTER reload should actually be findable
    q = vectors2[0]
    ids, _ = loaded.search(q, k=1, nprobe=loaded.nlist)  # probe everything, avoid a routing miss
    assert n_first in ids, "vector added after reload should be searchable"

    print(f"PASS: total_vectors after reload+more adds = {loaded.total_vectors()} "
          f"(expected {n_first + n_second}), post-reload insert is searchable")
    _clean()


if __name__ == "__main__":
    test_save_then_load_gives_identical_search_results()
    test_loaded_index_reports_correct_total_vectors_and_memory()
    test_can_keep_adding_after_load()
    print("\nall IVF+PQ persistence tests passed")
