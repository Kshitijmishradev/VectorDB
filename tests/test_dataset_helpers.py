import numpy as np
import pytest

from benchmarks.datasets import iter_rows, validate_sift_file
from benchmarks.bench_sift1m import _exact_subset_truth, _mark_selected
from vectordb.brute_force import BruteForceIndex
from demo.clip_tools import normalize_rows


def test_small_generated_hdf5_fixture(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train", data=np.zeros((12, 8), np.float32))
        handle.create_dataset("test", data=np.zeros((3, 8), np.float32))
        handle.create_dataset("neighbors", data=np.zeros((3, 10), np.int64))
    with h5py.File(path, "r") as handle:
        train, tests, neighbors = validate_sift_file(handle)
        batches = list(iter_rows(train, 5))
        assert train.shape == (12, 8)
        assert tests.shape == (3, 8)
        assert neighbors.shape == (3, 10)
        assert [len(batch) for _, batch in batches] == [5, 5, 2]


def test_small_generated_image_fixture(tmp_path):
    image_module = pytest.importorskip("PIL.Image")
    image_path = tmp_path / "tiny.png"
    image_module.new("RGB", (8, 8), (220, 30, 50)).save(image_path)
    with image_module.open(image_path) as image:
        assert image.size == (8, 8)
    normalized = normalize_rows(np.asarray([[3.0, 4.0]], dtype=np.float32))
    assert np.allclose(normalized, [[0.6, 0.8]])


def test_subset_ground_truth_matches_brute_force():
    rng = np.random.default_rng(8)
    vectors = rng.random((73, 16), dtype=np.float32)
    queries = rng.random((4, 16), dtype=np.float32)
    brute = BruteForceIndex(16, metric="l2")
    brute.add(vectors)
    expected = np.stack([brute.search(query, k=10)[0] for query in queries])
    actual = _exact_subset_truth(vectors, len(vectors), queries, batch_size=17)
    np.testing.assert_array_equal(actual, expected)


def test_hnsw_selection_uses_p95_then_memory_then_build():
    rows = [
        {"engine": "hnsw", "recall_at_10": 0.96, "p95_ms": 4.0,
         "resident_index_bytes": 200, "build_time_s": 2.0},
        {"engine": "hnsw", "recall_at_10": 0.97, "p95_ms": 3.0,
         "resident_index_bytes": 300, "build_time_s": 3.0},
        {"engine": "hnsw", "recall_at_10": 0.90, "p95_ms": 1.0,
         "resident_index_bytes": 100, "build_time_s": 1.0},
    ]
    selected = _mark_selected(rows)
    assert selected is rows[1]
    assert [row["selected"] for row in rows] == [False, True, False]
