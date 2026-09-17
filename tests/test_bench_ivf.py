import numpy as np

from bench_ivf import _streaming_ground_truth
from vectordb.brute_force import BruteForceIndex


def test_streaming_ground_truth_matches_brute_force():
    n = 2500
    dim = 32
    k = 10
    seed = 11
    queries = np.random.default_rng(seed + 2).random((4, dim), dtype=np.float32)

    vectors = np.random.default_rng(seed + 1).random((n, dim), dtype=np.float32)
    brute = BruteForceIndex(dim, metric="l2")
    brute.add(vectors)
    expected = np.stack([brute.search(query, k=k)[0] for query in queries])
    actual = _streaming_ground_truth(
        n, dim, queries, k, seed, batch_size=333, progress_every=n + 1)

    assert np.array_equal(
        np.sort(actual, axis=1), np.sort(expected, axis=1)), \
        f"streaming ground truth differs:\nactual={actual}\nexpected={expected}"
    print("PASS: streaming large-scale recall ground truth matches brute force")


if __name__ == "__main__":
    test_streaming_ground_truth_matches_brute_force()
