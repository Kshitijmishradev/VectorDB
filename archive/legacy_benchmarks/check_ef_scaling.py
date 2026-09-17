from benchmark import bench

# same n=10000 dataset, just raising ef_search to see if recall recovers
for ef in [50, 150, 300]:
    r = bench(10000, n_queries=20, ef_search=ef)
    print(f"ef_search={ef:4d}  hnsw_query={r['hnsw_query_ms']:7.3f}ms  "
          f"speedup={r['speedup']:5.2f}x  recall@10={r['recall@10']:.3f}")
