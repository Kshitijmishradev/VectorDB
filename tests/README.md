# Test suite

Run the complete suite from the project root:

```bash
python3 -m pytest
```

The files are intentionally separated by responsibility:

- `test_brute_force.py`, `test_hnsw.py`, and `test_pq.py` cover the
  foundational indexes.
- `test_hnsw_pq.py` covers compact adjacency, degree and reciprocal-edge
  invariants, scalar/batch equivalence, mmap reranking, honest graph memory,
  persistence, and continued insertion.
- `test_ivf_pq.py` covers IVF routing, recall, batching, and bounded memory.
- `test_ivf_pq_persistence.py` covers save/load and continued insertion.
- `test_ivf_pq_compaction.py` covers packed reads and delta re-compaction.
- `test_ivf_pq_rerank.py` covers raw-vector exact reranking.
- `test_ivf_pq_quality.py` covers residual PQ, candidate budgets, streaming
  top-N equivalence, accumulated training, metadata v3/v2 compatibility, and
  residual compaction with delta inserts.
- `test_dataset_helpers.py` uses generated HDF5 and image fixtures, verifies
  limited-corpus exact ground truth and HNSW tuning selection, and never
  downloads benchmark data or model weights.
- `test_bench_ivf.py` verifies the benchmark's streaming ground truth.
- `test_api.py` covers the REST flow, restart recovery, and validation.
