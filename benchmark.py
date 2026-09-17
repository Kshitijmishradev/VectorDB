"""
Benchmark: brute force vs HNSW, at increasing dataset sizes.
Measures query latency and recall, the two numbers that matter.
"""
import time
import os
import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.hnsw import HNSWIndex


def bench(n, dim=64, n_queries=30, k=10, M=16, ef_construction=200, ef_search=None, seed=0):  # None = auto-scale with n
    rng = np.random.default_rng(seed)
    vectors = rng.random((n, dim)).astype(np.float32)
    queries = rng.random((n_queries, dim)).astype(np.float32)

    brute = BruteForceIndex(dim)
    brute.add(vectors)

    t0 = time.perf_counter()
    for q in queries:
        brute.search(q, k=k)
    brute_query_time = (time.perf_counter() - t0) / n_queries

    hnsw = HNSWIndex(dim, M=M, ef_construction=ef_construction)
    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        hnsw.add(i, v)
    hnsw_build_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    hits, total = 0, 0
    for q in queries:
        true_ids, _ = brute.search(q, k=k)
        got_ids, _ = hnsw.search(q, k=k, ef_search=ef_search)
        hits += len(set(true_ids.tolist()) & set(got_ids))
        total += k
    hnsw_query_time = (time.perf_counter() - t0) / n_queries
    recall = hits / total

    speedup = brute_query_time / hnsw_query_time
    return {
        "n": n,
        "brute_query_ms": brute_query_time * 1000,
        "hnsw_query_ms": hnsw_query_time * 1000,
        "hnsw_build_s": hnsw_build_time,
        "speedup": speedup,
        "recall@10": recall,
    }


if __name__ == "__main__":
    results = []
    for n in [500, 1000, 2000, 5000, 10000]:
        r = bench(n)
        results.append(r)
        print(f"n={r['n']:6d}  brute={r['brute_query_ms']:7.3f}ms  "
              f"hnsw={r['hnsw_query_ms']:7.3f}ms  speedup={r['speedup']:6.1f}x  "
              f"recall@10={r['recall@10']:.3f}  hnsw_build={r['hnsw_build_s']:.1f}s")

    import json
    os.makedirs("results", exist_ok=True)
    with open("results/benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to results/benchmark_results.json")
