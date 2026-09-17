# Test suite

Run the complete suite from the project root:

```bash
python3 -m pytest
```

The files are intentionally separated by responsibility:

- `test_brute_force.py`, `test_hnsw.py`, `test_pq.py`, and
  `test_hnsw_pq.py` cover the foundational indexes.
- `test_ivf_pq.py` covers IVF routing, recall, batching, and bounded memory.
- `test_ivf_pq_persistence.py` covers save/load and continued insertion.
- `test_ivf_pq_compaction.py` covers packed reads and delta re-compaction.
- `test_ivf_pq_rerank.py` covers raw-vector exact reranking.
- `test_bench_ivf.py` verifies the benchmark's streaming ground truth.
- `test_api.py` covers the REST flow, restart recovery, and validation.

