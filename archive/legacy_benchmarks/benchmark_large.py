"""Same benchmark, pushed to larger n to find where HNSW actually crosses
over brute force. This is the scale where the O(log n) vs O(n) difference
should start showing up now that per-step overhead is fixed."""
from benchmark import bench
import json

if __name__ == "__main__":
    results = []
    for n in [20000, 50000, 100000]:
        r = bench(n, n_queries=20, ef_search=50)
        results.append(r)
        print(f"n={r['n']:7d}  brute={r['brute_query_ms']:8.3f}ms  "
              f"hnsw={r['hnsw_query_ms']:8.3f}ms  speedup={r['speedup']:6.2f}x  "
              f"recall@10={r['recall@10']:.3f}  hnsw_build={r['hnsw_build_s']:.1f}s")

    with open("benchmark_results_large.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved to benchmark_results_large.json")
