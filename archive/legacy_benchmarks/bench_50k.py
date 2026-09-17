from benchmark import bench
import json, time

t0 = time.time()
r = bench(50000, n_queries=15, ef_search=50)
print(f"n={r['n']:7d}  brute={r['brute_query_ms']:8.3f}ms  "
      f"hnsw={r['hnsw_query_ms']:8.3f}ms  speedup={r['speedup']:6.2f}x  "
      f"recall@10={r['recall@10']:.3f}  hnsw_build={r['hnsw_build_s']:.1f}s  "
      f"total_wall={time.time()-t0:.1f}s")
with open("benchmark_50k.json", "w") as f:
    json.dump(r, f, indent=2)
