import numpy as np
from vectordb.brute_force import BruteForceIndex


def test_finds_exact_match():
    """If the query vector is literally in the dataset, it must come back as
    result #1 with distance ~0. Simplest possible correctness check."""
    idx = BruteForceIndex(dim=8, metric="cosine")
    vectors = np.random.rand(100, 8).astype(np.float32)
    idx.add(vectors)

    query = vectors[42].copy()
    ids, dists = idx.search(query, k=5)

    assert ids[0] == 42, f"expected id 42 first, got {ids[0]}"
    assert dists[0] < 1e-5, f"expected ~0 distance to itself, got {dists[0]}"
    print("PASS: exact match test")


def test_matches_naive_python_loop():
    """Cross-check our vectorized numpy code against the dumbest possible
    correct implementation: a plain python for-loop computing distances one
    at a time. If these ever disagree, the numpy version has a bug."""
    np.random.seed(1)
    idx = BruteForceIndex(dim=16, metric="l2")
    vectors = np.random.rand(200, 16).astype(np.float32)
    idx.add(vectors)

    query = np.random.rand(16).astype(np.float32)

    # naive ground truth, computed the slow obvious way
    naive_dists = [np.sum((v - query) ** 2) for v in vectors]
    naive_top5 = np.argsort(naive_dists)[:5]

    ids, dists = idx.search(query, k=5)

    assert list(ids) == list(naive_top5), f"{list(ids)} != {list(naive_top5)}"
    print("PASS: matches naive loop")


def test_k_larger_than_dataset():
    """Edge case: asking for more neighbors than exist shouldn't crash."""
    idx = BruteForceIndex(dim=4)
    idx.add(np.random.rand(3, 4).astype(np.float32))
    ids, dists = idx.search(np.random.rand(4).astype(np.float32), k=100)
    assert len(ids) == 3
    print("PASS: k larger than dataset")


def test_recall_at_k_is_100_percent_against_itself():
    """Sanity check on the metric we'll use everywhere later: brute force
    compared to brute force must be perfect recall, always. If this isn't
    1.0, the recall@k measuring code itself is broken."""
    np.random.seed(2)
    idx = BruteForceIndex(dim=32, metric="cosine")
    vectors = np.random.rand(500, 32).astype(np.float32)
    idx.add(vectors)

    hits = 0
    n_queries = 20
    k = 10
    for _ in range(n_queries):
        q = np.random.rand(32).astype(np.float32)
        true_ids, _ = idx.search(q, k=k)
        got_ids, _ = idx.search(q, k=k)  # same index, should be identical
        hits += len(set(true_ids) & set(got_ids))

    recall = hits / (n_queries * k)
    assert recall == 1.0, f"expected recall 1.0, got {recall}"
    print(f"PASS: recall@{k} = {recall}")


if __name__ == "__main__":
    test_finds_exact_match()
    test_matches_naive_python_loop()
    test_k_larger_than_dataset()
    test_recall_at_k_is_100_percent_against_itself()
    print("\nall tests passed")
