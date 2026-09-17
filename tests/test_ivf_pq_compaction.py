import builtins
import os
import shutil
from unittest.mock import patch

import numpy as np

from vectordb.ivf_pq import IVFPQIndex


STORAGE = "/tmp/ivf_pq_compaction_test_storage"


def _clean(index=None):
    if index is not None:
        index.close()
    if os.path.exists(STORAGE):
        shutil.rmtree(STORAGE)


def _searches(index, queries):
    return [index.search(query, k=10, nprobe=8) for query in queries]


def test_compaction_preserves_search_and_eliminates_cluster_opens():
    _clean()
    rng = np.random.default_rng(23)
    vectors = rng.random((3000, 32), dtype=np.float32)
    queries = rng.random((12, 32), dtype=np.float32)
    index = IVFPQIndex(32, nlist=30, pq_m=8, storage_dir=STORAGE, seed=4)
    index.train(vectors[:1500], n_iters=5)
    index.add_batch(vectors, np.arange(len(vectors), dtype=np.int64))

    before = _searches(index, queries)
    compacted = index.compact()
    after = _searches(index, queries)

    assert index.is_compacted
    assert compacted["vectors"] == len(vectors)
    assert compacted["bytes"] == len(vectors) * (8 + index.pq.m)
    assert index.disk_bytes() == compacted["bytes"]
    assert all(not os.path.exists(index._cluster_path(c)) for c in range(index.nlist))

    for (before_ids, before_dists), (after_ids, after_dists) in zip(before, after):
        assert before_ids == after_ids
        assert np.allclose(before_dists, after_dists)

    real_open = builtins.open
    posting_opens = []

    def counting_open(path, *args, **kwargs):
        if str(path).endswith(".bin"):
            posting_opens.append(str(path))
        return real_open(path, *args, **kwargs)

    with patch("builtins.open", side_effect=counting_open):
        _searches(index, queries)
    assert posting_opens == [], f"compacted search opened posting files: {posting_opens}"

    print("PASS: compaction preserves results and compacted search opens zero posting files")
    _clean(index)


def test_compacted_save_load_delta_add_and_recompact():
    _clean()
    rng = np.random.default_rng(29)
    first = rng.random((1800, 32), dtype=np.float32)
    second = rng.random((200, 32), dtype=np.float32)
    queries = rng.random((8, 32), dtype=np.float32)

    index = IVFPQIndex(32, nlist=24, pq_m=8, storage_dir=STORAGE, seed=5)
    index.train(first[:1200], n_iters=5)
    index.add_batch(first, np.arange(len(first), dtype=np.int64))
    index.compact()
    packed_results = _searches(index, queries)
    index.close()

    loaded = IVFPQIndex.load(STORAGE)
    assert loaded.is_compacted
    loaded_results = _searches(loaded, queries)
    for (expected_ids, expected_dists), (actual_ids, actual_dists) in zip(
            packed_results, loaded_results):
        assert expected_ids == actual_ids
        assert np.allclose(expected_dists, actual_dists)

    second_ids = np.arange(len(first), len(first) + len(second), dtype=np.int64)
    loaded.add_batch(second, second_ids)
    loaded.save()
    assert loaded.total_vectors() == len(first) + len(second)
    assert any(os.path.exists(loaded._delta_path(c)) for c in range(loaded.nlist))

    # Search all clusters so a newly added vector cannot be missed by routing.
    found_ids, _ = loaded.search(second[0], k=1, nprobe=loaded.nlist)
    assert int(second_ids[0]) in found_ids

    loaded.compact()
    assert all(not os.path.exists(loaded._delta_path(c)) for c in range(loaded.nlist))
    assert loaded.disk_bytes() == (len(first) + len(second)) * (8 + loaded.pq.m)
    loaded.close()

    reloaded = IVFPQIndex.load(STORAGE)
    assert reloaded.is_compacted
    assert reloaded.total_vectors() == len(first) + len(second)
    found_ids, _ = reloaded.search(second[0], k=1, nprobe=reloaded.nlist)
    assert int(second_ids[0]) in found_ids

    print("PASS: compacted persistence, delta inserts, and re-compaction work")
    _clean(reloaded)


if __name__ == "__main__":
    test_compaction_preserves_search_and_eliminates_cluster_opens()
    test_compacted_save_load_delta_add_and_recompact()
    print("\nall IVF+PQ compaction tests passed")
