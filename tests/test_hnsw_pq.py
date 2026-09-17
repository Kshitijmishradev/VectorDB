import time
import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.hnsw_pq import HNSWPQIndex


def test_tiny_sanity_check():
    np.random.seed(0)
    dim = 32
    vectors = np.random.rand(10, dim).astype(np.float32)

    idx = HNSWPQIndex(dim, M=8, ef_construction=50, pq_m=8, pq_k=8)
    idx.train(vectors)
    for i, v in enumerate(vectors):
        idx.add(i, v)

    correct = 0
    for i, q in enumerate(vectors):
        got_ids, _ = idx.search(q, k=1, ef_search=20)
        if got_ids[0] == i:
            correct += 1
    print(f"PASS (informational): {correct}/10 exact self-matches through compression")
    assert correct >= 7, f"expected most self-matches to survive compression, got {correct}/10"


def test_recall_and_memory_vs_brute_force():
    np.random.seed(1)
    dim = 64
    n = 3000
    vectors = np.random.rand(n, dim).astype(np.float32)
    queries = np.random.rand(30, dim).astype(np.float32)
    k = 10

    brute = BruteForceIndex(dim, metric="l2")
    brute.add(vectors)
    brute_bytes = brute.vectors.nbytes

    idx = HNSWPQIndex(dim, M=16, ef_construction=200, pq_m=16, pq_k=256)
    idx.train(vectors)
    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        idx.add(i, v)
    build_time = time.perf_counter() - t0

    hits, total = 0, 0
    for q in queries:
        true_ids, _ = brute.search(q, k=k)
        got_ids, _ = idx.search(q, k=k)
        hits += len(set(true_ids.tolist()) & set(got_ids))
        total += k
    recall = hits / total

    mem_ratio = brute_bytes / idx.memory_bytes()
    print(f"PASS (informational): n={n} build={build_time:.1f}s recall@{k}={recall:.3f} "
          f"memory: brute={brute_bytes/1024:.1f}KB combined={idx.memory_bytes()/1024:.1f}KB "
          f"ratio={mem_ratio:.1f}x")

    assert recall >= 0.4, f"recall too low: {recall:.3f}"
    assert mem_ratio > 5, f"expected real memory savings, got only {mem_ratio:.1f}x"


def test_train_before_add_is_enforced():
    idx = HNSWPQIndex(dim=16, pq_m=4, pq_k=8)
    try:
        idx.add(0, np.random.rand(16).astype(np.float32))
        assert False, "expected an assertion error for add() before train()"
    except AssertionError as e:
        assert "train" in str(e)
        print("PASS: add() before train() correctly rejected")


if __name__ == "__main__":
    test_tiny_sanity_check()
    test_recall_and_memory_vs_brute_force()
    test_train_before_add_is_enforced()
    print("\nall HNSW+PQ tests passed")
