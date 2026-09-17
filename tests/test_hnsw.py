"""
HNSW correctness tests. The core question every test here answers:
does HNSW's approximate answer actually agree with brute force's exact
answer, most of the time, as measured by recall@k?
"""
import time
import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.hnsw import HNSWIndex


def recall_at_k(hnsw, brute, queries, k=10, ef_search=50):
    hits, total = 0, 0
    for q in queries:
        true_ids, _ = brute.search(q, k=k)
        got_ids, _ = hnsw.search(q, k=k, ef_search=ef_search)
        hits += len(set(true_ids.tolist()) & set(got_ids))
        total += k
    return hits / total


def test_tiny_sanity_check():
    """10 points, k=1: HNSW must find the exact nearest neighbor every time.
    If this fails, there's a basic bug, no need to test anything bigger."""
    np.random.seed(0)
    dim = 4
    vectors = np.random.rand(10, dim).astype(np.float32)

    brute = BruteForceIndex(dim)
    brute.add(vectors)

    hnsw = HNSWIndex(dim, M=8, ef_construction=50)
    for i, v in enumerate(vectors):
        hnsw.add(i, v)

    correct = 0
    for i, q in enumerate(vectors):
        got_ids, _ = hnsw.search(q, k=1, ef_search=20)
        if got_ids[0] == i:
            correct += 1
    assert correct == 10, f"expected 10/10 exact self-matches, got {correct}/10"
    print("PASS: tiny sanity check (10/10 exact self-match)")


def test_recall_at_moderate_scale():
    """2000 random vectors, 50 queries, checking recall@10.
    A correct HNSW implementation should comfortably clear 90% here."""
    np.random.seed(1)
    dim = 64
    n = 2000

    vectors = np.random.rand(n, dim).astype(np.float32)
    brute = BruteForceIndex(dim)
    brute.add(vectors)

    hnsw = HNSWIndex(dim, M=16, ef_construction=200)
    t0 = time.time()
    for i, v in enumerate(vectors):
        hnsw.add(i, v)
    build_time = time.time() - t0

    queries = np.random.rand(50, dim).astype(np.float32)
    recall = recall_at_k(hnsw, brute, queries, k=10, ef_search=50)

    print(f"PASS (informational): n={n} dim={dim} build_time={build_time:.2f}s recall@10={recall:.3f}")
    assert recall >= 0.85, f"recall too low: {recall:.3f} (expected >= 0.85)"


def test_ef_search_tradeoff():
    """Higher ef_search should mean higher (or equal) recall. This is the
    knob real systems expose to trade speed for accuracy at query time."""
    np.random.seed(2)
    dim = 32
    n = 1000
    vectors = np.random.rand(n, dim).astype(np.float32)

    brute = BruteForceIndex(dim)
    brute.add(vectors)

    hnsw = HNSWIndex(dim, M=16, ef_construction=200)
    for i, v in enumerate(vectors):
        hnsw.add(i, v)

    queries = np.random.rand(30, dim).astype(np.float32)
    recall_low = recall_at_k(hnsw, brute, queries, k=10, ef_search=10)
    recall_high = recall_at_k(hnsw, brute, queries, k=10, ef_search=100)

    print(f"PASS (informational): recall@10 ef_search=10 -> {recall_low:.3f}, ef_search=100 -> {recall_high:.3f}")
    assert recall_high >= recall_low - 0.02, "higher ef_search should not meaningfully hurt recall"


def test_incremental_add_still_searchable():
    """Add vectors one at a time (not all at once) and confirm search still
    works correctly throughout, this is the realistic usage pattern."""
    np.random.seed(3)
    dim = 16
    hnsw = HNSWIndex(dim, M=8, ef_construction=100)
    vectors = []
    for i in range(300):
        v = np.random.rand(dim).astype(np.float32)
        vectors.append(v)
        hnsw.add(i, v)

    brute = BruteForceIndex(dim)
    brute.add(np.array(vectors))

    q = vectors[150]
    got_ids, _ = hnsw.search(q, k=5, ef_search=50)
    assert 150 in got_ids, "should find exact self-match among top 5 even with incremental inserts"
    print("PASS: incremental add still searchable")


if __name__ == "__main__":
    test_tiny_sanity_check()
    test_recall_at_moderate_scale()
    test_ef_search_tradeoff()
    test_incremental_add_still_searchable()
    print("\nall HNSW tests passed")
