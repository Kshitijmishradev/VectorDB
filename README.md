# vectordb

A vector database built entirely from scratch: no faiss, no hnswlib, no
sklearn, no external ANN library of any kind. Every algorithm here
(exact search, HNSW, product quantization, IVF, k-means) is implemented
directly in numpy, tested against real ground truth, and benchmarked
with real numbers, on real hardware, not simulated or assumed.

The goal wasn't just "make something that returns nearest neighbors."
It was to understand, at the level of actually writing the code, how
systems like Pinecone, Weaviate, Qdrant, and Chroma work internally, and
to prove that understanding with measured evidence rather than
citations. Every claim in this README and in [`RESULTS.md`](./RESULTS.md)
is backed by a test or a benchmark you can re-run yourself.

## What it actually does

Store high-dimensional vectors, and given a query vector, find its k
nearest neighbors, fast, at scale, without needing the whole dataset in
memory. This is the core operation behind semantic search, recommendation
systems, RAG pipelines, and image similarity search.

```
POST /collections/products {"dim": 64, "nlist": 50}
POST /collections/products/train {"vectors": [[...], ...]}
POST /collections/products/vectors {"ids": [1,2,3], "vectors": [[...], [...], [...]]}
POST /collections/products/search {"vector": [...], "k": 5}
-> {"ids": [42, 17, 891, 3, 256], "distances": [0.12, 0.31, 0.44, 0.51, 0.58]}
```

## The architecture, and why each piece exists

Built in this order, each stage solving a specific limitation of the one
before it:

| Stage | File | Solves | Cost |
|---|---|---|---|
| **Brute force** | `brute_force.py` | Ground truth: exact, always correct | O(n) per query, no compression |
| **HNSW** | `hnsw.py` | O(log n)-ish search via a multi-layer navigable graph | Still stores full float vectors in RAM |
| **Product Quantization** | `pq.py` | 8-32x memory compression via learned codebooks | Lossy; recall/compression is a real tradeoff |
| **HNSW + PQ** | `hnsw_pq.py` | Combines graph speed with compressed storage | Build is slower (more work per insertion) |
| **IVF + PQ** | `ivf_pq.py` | Disk-backed storage; RAM only holds a tiny routing index | Coarser-grained approximation (`nprobe` tradeoff) |

**Brute force** computes the distance from a query to every stored
vector and returns the smallest k. Always exactly correct, and this is
what every approximate method's answers get checked against
(`recall@10`, used throughout this project, means "what fraction of
brute force's true top-10 did the approximate method actually find").

**HNSW** (Hierarchical Navigable Small World graphs) builds a multi-layer
graph where each vector links to its nearest neighbors, with higher
layers acting as a coarse "highway system" for fast approximate
traversal. Search greedily hops toward the query, funneling down through
layers. `ef_search` is the speed/recall dial: search more candidates,
better recall, slower.

**Product Quantization** compresses each vector by splitting it into
subspaces, running k-means per subspace to find representative
"centroids," and storing which centroid each subspace is closest to (a
single byte) instead of the real floats. Distance to a compressed vector
is computed via a precomputed lookup table (Asymmetric Distance
Computation), not by decompressing anything, so it's both smaller and
fast.

**HNSW + PQ** stores a PQ code (a few bytes) at each graph node instead
of a full vector, this is what Faiss's `IndexIVFPQ` and similar
production systems do: graph speed, compressed storage, one shared
distance table per query reused across the whole traversal.

**IVF + PQ** is the one that actually answers "how do you get past what
fits in RAM": the dataset is coarse-clustered into `nlist` groups
(k-means on full vectors), only the centroids stay in memory (a few MB,
independent of dataset size), and every actual vector is PQ-compressed.
Ingestion initially uses per-cluster files, then `compact()` packs them into
one offset-indexed, memory-mapped `postings.bin`. A query compares against all
centroids, picks the `nprobe` nearest clusters, and slices those lists from the
mapping without opening one file per cluster. New inserts use small delta
files until the next compaction. An optional raw-vector sidecar enables exact
reranking of a small PQ shortlist: IVF+PQ narrows millions of vectors to (for
example) 100 candidates, then exact L2 chooses the final 10. This is
architecturally close to Microsoft's SPANN and Faiss's `IndexIVFPQ`.

## The engineering, not just the algorithms

This is the part that actually took the most effort, and it's the part
most "from scratch" projects skip.

### The same bug, found and fixed seven separate times

Every serious performance problem in this project traced back to one
root cause: a Python-level `for` loop where a single vectorized numpy
call would do the same work at C speed. This wasn't found once and
generalized, it was independently rediscovered in five different files,
each time root-caused and fixed with real before/after measurements
rather than a guessed threshold change:

1. **HNSW's distance batching** — per-neighbor Python loop → `np.stack` + one matmul
2. **HNSW's insertion path** — dict-of-vectors + list comprehension → a preallocated numpy matrix with internal integer indices (build time at n=10,000: 191.9s → 70.0s → 13.2s across two separate fixes)
3. **PQ's `asymmetric_distances`/`decode`** — per-subspace Python loop → `table[idx, codes]` fancy-indexing (this alone made combined HNSW+PQ build 1.87x faster and query 2.4x faster)
4. **IVF's `_read_cluster`** — per-record `struct.unpack_from` loop → one `np.frombuffer` call with a structured dtype, caught by code review before it ever caused a slow benchmark
5. **IVF and PQ's own k-means training** — an `(n, k, dim)` broadcast distance array, fine for PQ's small subspaces, but at IVF's scale (`nlist` in the thousands, full vector dimensionality) this **actually OOM-killed the process** at ~13GB for one intermediate array. Fixed with the expanded distance identity (`|a-b|² = |a|² - 2a·b + |b|²`) computed in row-chunks, bounding peak memory to `O(batch_size × k)` instead of `O(n × k)`.
6. **IVF ingestion** — the streaming benchmark generated input in batches but still routed, PQ-encoded, and opened a posting-list file once per vector. `add_batch()` now routes with bounded BLAS matrix multiplications, encodes PQ in chunks, groups records by cluster, and writes once per non-empty cluster. At `nlist=12,649`, a 100,000-vector hot-path benchmark improved from roughly 2,073 to 55,872 vectors/sec.
7. **IVF query reads** — probing 1,000 clusters meant 1,000 separate `open`/`read`/`close` cycles for every query. `compact()` now writes one immutable posting segment plus a 20,001-entry offset table and keeps the segment memory-mapped. A 200,000-vector comparison reduced average query time from 2.079ms to 0.910ms (2.29x) while returning identical results and opening zero posting files during compacted search.
8. **Exact shortlist reranking** — optional `store_full_vectors=True` appends each float32 vector to a memory-mapped sidecar and stores its row number in the posting record. `search(..., rerank=100)` performs ordinary IVF+PQ candidate selection, reads only those 100 original vectors, and reorders them by exact L2. On a 200,000-vector sweep at `nprobe=256`, recall@10 improved from 0.480 to 0.787 with a 100-vector shortlist, while median latency stayed around 3.1ms.
9. **Apple GPU ingestion routing** — `routing_backend="mlx"` moves the coarse vector-to-centroid matrix multiplication to the Apple Silicon Metal GPU. It does not change PQ encoding, posting writes, search, or the index format. At the 100,000 x 20,000 x 64 routing shape, the isolated benchmark measured 1.644s for NumPy vs 0.543s for MLX (3.03x) with identical assignments.

### Honest, measured tradeoffs, not hand-waved ones

Every tunable parameter in this project (`ef_search`, PQ's `m`, IVF's
`nprobe`) has a real measured recall-vs-speed curve behind it, not a
default picked by feel:

- `ef_search` fixed at 50 caused recall to collapse from 0.997 (n=1,000) to 0.593 (n=50,000). Fixed with an empirically-calibrated scaling formula (`max(30, M·log₂(n))`).
- PQ's `m=8` gives 32x compression but recall@10=0.393; `m=16` gives 16x compression and recall@10=0.680; `m=32` gives 8x and 0.893. All three measured, not estimated.
- IVF's `nprobe` sweep (1 → 5 → 20) showed recall climbing 0.200 → 0.470 → 0.665, monotonically, confirming the routing logic is actually working correctly, not just "roughly okay."

### A cross-platform correctness bug, caught by not trusting a suspicious number

A benchmark run on the author's Mac reported a process memory reading
over a billion (in a field labeled KB). Rather than dismiss it as a
fluke, it was root-caused: Python's `resource.getrusage().ru_maxrss` has
OS-dependent units (kilobytes on Linux, bytes on macOS, and inconsistent
in practice across different measurement scales on the same machine).
Verified directly against a live process using `ps -o rss=` before
trusting any fix, then replaced the ambiguous field entirely with a
`ps`-based reading that's unambiguous on both platforms.

### A benchmark that scales the way the database does

The original large-scale benchmark script pre-generated its entire
synthetic test dataset in one array before inserting anything, at
n=1,000,000 that's 256MB just for test input, at n=1,000,000,000 it's
256GB, impossible on a laptop regardless of how good the actual
database's memory story is. Rewritten to stream the dataset in batches,
discarding each batch immediately after insertion, so the *test
harness's* memory stays flat too, not just the database's. Verified the
fix with a direct measurement: RSS grew by exactly one batch's worth of
data during a streamed 300,000-vector build, not the full dataset's
worth.

### Persistence: the database survives a restart

`IVFPQIndex.save()`/`.load()` persist the routing index (centroids + PQ
codebooks + cluster sizes + packed offsets) to a small file; `postings.bin`
holds the immutable packed base and per-cluster delta files hold later
inserts. Tested the following real claims, not
just "does it not crash": a reloaded index returns byte-identical search
results to before the save; bookkeeping (`total_vectors`,
`resident_memory_bytes`) reports correctly after reload; and a reloaded
index can keep accepting new inserts, with a vector added after reload
actually being findable.

### A real REST API, with correctness proven under an actual simulated restart

`api.py` wraps `IVFPQIndex` in a FastAPI service (`create` / `train` /
`add` / `compact` / `search` / `stats` per named "collection", plus auto-generated
interactive docs at `/docs`). The persistence work is what makes this
safe to run as an actual service: every mutation auto-saves, and server
startup automatically reloads every collection found on disk. This was
tested by literally wiping the in-memory collection registry mid-test
(simulating a real process crash) and confirming a fresh client
reloads the collection with the right vector count and correct search
results, not just checking the save/load functions in isolation.

## Real, measured results

The headline number, from the author's own MacBook, not a cloud sandbox:

```
n = 1,000,000 vectors, dim = 64
build:            226s     (4,425 vectors/sec)
query latency:    7.5ms
recall@10:        0.37     (stable across every scale tested, 5,000 to 1,000,000)
resident memory:  1.06 MB  <- for one million vectors
disk used:        22.9 MB
```

One megabyte of RAM for a million searchable vectors. That number stays
essentially flat at any scale, because it depends on `nlist` (which
grows as `√n`), not on the number of vectors stored. Full numbered
session log, every benchmark, every bug, every before/after measurement,
is in [`RESULTS.md`](./RESULTS.md).

## What's honestly still missing

This project does not claim to be a production-grade database. Known
gaps, listed rather than glossed over:

- **No delete/update.** Every posting list is append-only.
- **Reranking costs disk.** It is opt-in because retaining float32 vectors adds 256 bytes per 64-dimensional vector. At 25M vectors, the final raw-vector-plus-posting index is about 6.71 GiB instead of about 572 MiB for PQ-only storage.
- **No concurrency safety.** FastAPI dispatches synchronous endpoints to a real OS thread pool by default, and shared state (cluster file writes, in-memory counters) has no locking. Two simultaneous writes to the same collection can race.
- **No auth, no streaming uploads, no metadata filtering.** Request-level
  ingestion is batched, but very large uploads still need a streaming API.

## Project structure

```
vectordb/
  brute_force.py     exact search, the ground truth
  hnsw.py             graph-based ANN search
  pq.py               product quantization + PQFlatIndex
  hnsw_pq.py           HNSW graph + PQ-compressed storage
  ivf_pq.py            disk-backed IVF + PQ, with save()/load()
  mlx_routing.py        optional Apple Silicon GPU ingestion router
  api.py               FastAPI REST wrapper (collections, train, add, search)

tests/                 complete correctness and API integration suite
benchmark.py            HNSW vs brute force benchmark sweep
bench_ivf.py            large-scale IVF+PQ benchmark (streaming, no memory cliff)
bench_tuning.py         measured nprobe latency/recall sweep
bench_gpu_routing.py    isolated NumPy CPU vs MLX GPU routing benchmark
run_api.py              start the REST API as a real local server
RESULTS.md              the full, honest, numbered log of every session
results/                saved JSON benchmark evidence
archive/                superseded one-off benchmark drivers
```

## Running it yourself

```bash
pip install -r requirements-dev.txt

# correctness, every module and integration path
python3 -m pytest

# one file directly, when iterating on a specific feature
python3 -m tests.test_ivf_pq_rerank

# benchmarks
python3 benchmark.py
python3 bench_ivf.py 1000000      # takes a few minutes; no upper limit besides time and disk
python3 bench_ivf.py 200000 --compare-unpacked-query  # measure packed read speedup
python3 bench_tuning.py 200000     # choose nprobe from latency + recall, not a percentage
python3 bench_gpu_routing.py --vectors 100000 --nlist 20000 \
  --json results/gpu_routing_benchmark.json

# Measure the nprobe + rerank recall/latency curve before a large run
python3 bench_tuning.py 200000 --nprobes 64,128,256 \
  --rerank-values 0,50,100,200

# 25M preflight, then a memory-bounded run with durable JSON results
python3 bench_ivf.py 25000000 --dry-run
python3 bench_ivf.py 25000000 --progress-every 1000000 \
  --json results/benchmark_25m.json

# Faster approximate training (validate recall with bench_tuning.py first)
python3 bench_ivf.py 25000000 --train-minibatch-size 200000 \
  --pq-train-size 200000 --progress-every 1000000 \
  --json results/benchmark_25m_fast.json

# Apple Silicon build routing + exact top-100 reranking. Final index is ~6.71 GiB.
python3 bench_ivf.py 25000000 --train-minibatch-size 200000 \
  --pq-train-size 200000 --routing-backend mlx --rerank 100 \
  --progress-every 1000000 --recall-queries 3 --compare-unpacked-query \
  --keep-storage \
  --json results/benchmark_25m_rerank_mlx.json

# Compare the fast-training recall/latency curve on a manageable sample first
python3 bench_tuning.py 200000 --train-minibatch-size 200000 --pq-train-size 200000

# the API, as an actual running server
python3 run_api.py
# then open http://localhost:8000/docs
```

## Why this project exists

Built as a portfolio project to demonstrate genuine understanding of how
vector search systems work, not just the ability to call a library.
Every design decision above was driven by hitting a real limitation
(too slow, too much memory, doesn't survive a restart, doesn't scale
past RAM) and fixing it with a specific, measured, explainable change,
the same way a real engineering team would.
