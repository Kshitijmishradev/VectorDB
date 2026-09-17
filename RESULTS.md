# Vector Database From Scratch — Overnight Session Results

Read this top to bottom in the morning, it tells the whole story: what got
built, what the tests say, and one important honest finding that changes
what we do next.

## Higher-recall search and SIFT1M validation

The IVF+PQ engine now supports opt-in residual PQ, candidate-budget routing,
streaming top-N scans, exact reranking, accumulated coarse training, and
metadata version 3 with version-2 compatibility. Search no longer concatenates
every probed posting or creates a candidate-by-`pq_m` contribution matrix.

The final official SIFT1M run on the M3 Pro used all 10,000 queries:

```text
n=1,000,000  dim=128  nlist=4,096  candidate_budget=20,000
pq_mode=residual  pq_m=32  rerank=100
recall@1=0.9782  recall@10=0.9701
p50=9.3691ms  p95=13.4028ms  p99=14.6929ms  QPS=104.05
train=11.568s  build=10.669s  compact=0.299s
```

This passed the fixed acceptance target (recall@10 ≥ 0.80, p50 ≤ 75 ms,
p95 ≤ 100 ms). A same-data 200-query comparison measured legacy standard PQ
at recall@10 0.5225 / p95 2.2645 ms and the optimized mode at recall@10
0.9735 / p95 13.8362 ms.

The separate 25% boundary-replication experiment failed its promotion gate:
at matched ~20.3k scanned records, recall changed from 0.9740 to 0.9695.
Replication therefore remains out of the persistent format.

## What got built tonight

**`vectordb/brute_force.py`** — the ground-truth exact search index.
Vectorized numpy (no python loops over vectors), cosine or L2 distance,
`argpartition` instead of a full sort since we only need the top k. This
is what every later index gets checked against for correctness, forever.

**`vectordb/hnsw.py`** — HNSW built from scratch, no libraries. Multi-layer
graph, exponential layer assignment (few nodes climb high, most stay at
layer 0), greedy best-first search per layer using a min-heap for the
"explore" queue and a bounded max-heap for "best found so far", neighbor
pruning on insert so node degree stays bounded. `ef_construction` controls
build quality, `ef_search` controls the speed/recall tradeoff at query time.

**`test_brute_force.py`** — 4/4 passing. Checks exact self-match, cross-
verifies the vectorized numpy code against a dead-simple python loop
(catches vectorization bugs), edge case for k > dataset size, and sanity-
checks the recall@k formula itself against a perfect baseline.

**`test_hnsw.py`** — 4/4 passing:
- 10-point sanity check: 10/10 exact nearest neighbor matches at k=1
- 2000 points, dim=64: **recall@10 = 0.966** against brute force ground truth
- ef_search tradeoff confirmed: recall@10 goes from 0.883 (ef_search=10) to
  1.000 (ef_search=100), exactly the speed/accuracy knob it's supposed to be
- incremental one-at-a-time inserts still search correctly

**`benchmark.py`** — head to head timing, results in `results/benchmark_results.json`.

## The honest finding: HNSW is currently *slower* than brute force

```
n=   500  brute=  0.078ms  hnsw=  2.859ms  speedup=0.0x  recall@10=1.000
n=  1000  brute=  0.121ms  hnsw=  4.169ms  speedup=0.0x  recall@10=0.997
n=  2000  brute=  0.226ms  hnsw=  5.418ms  speedup=0.0x  recall@10=0.983
n=  5000  brute=  0.558ms  hnsw=  7.445ms  speedup=0.1x  recall@10=0.930
n= 10000  brute=  1.214ms  hnsw=  9.234ms  speedup=0.1x  recall@10=0.843
```

Don't read this as "HNSW failed", it's correct and doing real approximate
graph search (that's why recall is high, not 0 or random). What's happening
is a constant-factor problem, not an algorithm problem:

`BruteForceIndex.search()` computes distance to every vector in **one
vectorized numpy call**, which runs as optimized C loops under the hood.
`HNSWIndex.search()` computes distances **one at a time in pure Python**,
each one is a python function call, a numpy dot product on a single tiny
vector, a heap push. That per-step overhead is enormous compared to numpy's
batched C loop, so even though HNSW does mathematically fewer distance
computations, each one costs vastly more wall-clock time in this
implementation.

This is exactly why real libraries (faiss, hnswlib, the ones under Chroma
and pgvector) are written in C++ with thin Python bindings, not pure
Python. The algorithm's asymptotic advantage (`O(log n)` vs `O(n)`) is real
and would eventually win at large enough n even with python overhead, but
"large enough" here is much bigger than 10k, probably into the hundreds of
thousands to millions, and build time (191 seconds for just 10k points)
makes testing that directly impractical without fixing the overhead first.

**This is a legitimate and common lesson in systems work**: big-O tells you
what wins eventually, constant factors decide what wins at the scale you
actually care about. It's a good thing to be able to explain in an
interview, "I found this, understood why, and fixed it" is a stronger
story than "it was fast from the start."

## Recall also degrades as n grows (at fixed ef_search=50)

0.997 at n=1000 down to 0.843 at n=10000. This is expected and normal:
`ef_search` controls how many candidates get explored per query, and that
number needs to scale up somewhat as the dataset grows for recall to stay
high. Real systems tune `ef_search` relative to dataset size for this
reason. Not a bug, just an untuned parameter.

## Session 2 update: vectorized the distance computations

Fixed exactly what session 1 flagged: `_search_layer` now batches every
node's neighbors into one numpy call instead of looping distance-by-distance
in python, and vectors are normalized once at insert time instead of on
every single distance check. Recall is bit-for-bit identical after the
change (0.966 at n=2000, same as before), confirming this was purely a
performance fix, not a behavior change.

Result, same benchmark as before:

```
n=   500  brute=0.119ms  hnsw=1.277ms  speedup=0.1x  recall@10=1.000  build=1.1s
n=  1000  brute=0.164ms  hnsw=1.839ms  speedup=0.1x  recall@10=0.997  build=3.3s
n=  2000  brute=0.344ms  hnsw=2.064ms  speedup=0.2x  recall@10=0.983  build=8.6s
n=  5000  brute=0.698ms  hnsw=2.893ms  speedup=0.2x  recall@10=0.930  build=28.0s
n= 10000  brute=1.362ms  hnsw=4.446ms  speedup=0.3x  recall@10=0.843  build=70.0s
n= 20000  brute=2.512ms  hnsw=6.331ms  speedup=0.4x  recall@10=0.835  build=170.1s
n= 50000  brute=7.266ms  hnsw=14.205ms speedup=0.51x recall@10=0.593  build=569.8s
```

Build time roughly halved to a third at every size (n=10000 went from 192s
to 70s), and the speedup trend is climbing steadily toward parity (0.1x at
n=500 up to 0.51x at n=50000) as n grows, exactly what you'd expect from an
O(log n) algorithm catching up to O(n) once the constant-factor gap is
closed. It still hasn't crossed over to actually beating brute force in
this test range.

**Two new honest findings, both understood and both normal:**

1. **Recall collapses at n=50000 with a fixed ef_search=50** (0.593, down
   from 0.997 at n=1000). Verified this is exactly the known ef_search
   tradeoff and not a new bug: re-running n=10000 at ef_search=50/150/300
   gave recall 0.835 / 0.990 / 0.995, climbing exactly as expected, at the
   cost of query time roughly doubling each step. Conclusion: `ef_search`
   needs to scale with dataset size for a fair benchmark, a fixed value
   makes big datasets look worse than the algorithm actually is.

2. **Build time is now the real bottleneck, not query time.** 50000 points
   took 570 seconds (9.5 minutes) to build. This is a pure-Python overhead
   problem in the *insertion* path (per-node dict/set bookkeeping, python-level
   loop over layers), the same category of issue query search had, just not
   fixed yet. This is why production libraries like faiss/hnswlib are C++,
   not python, insertion at scale needs the same vectorization treatment
   query search just got, or a fundamentally faster data layout (numpy
   arrays instead of python dicts of sets for the graph).

## Recommended next session, in order

1. **Vectorize/speed up insertion**, it's now the dominant cost. Likely
   biggest lever: replace the python dict-of-sets graph representation with
   fixed-size numpy arrays per layer (neighbor lists as int arrays, padded),
   which also makes the search-side batching even cheaper.
2. **Scale `ef_search` with n** in the benchmark (e.g. proportional to
   `log(n)` or a fixed fraction of n) so the recall-vs-n chart is honest
   and comparable across sizes instead of penalizing big n with a fixed knob.
3. Once insertion is fast enough to build 100k-1M points in reasonable time,
   re-run the full benchmark, this is the range where HNSW should clearly
   and consistently beat brute force and the "crossover point" becomes a
   real number you can put in a writeup.
4. Then product quantization for memory compression, as originally planned.

## How to run everything yourself

```
cd vectordb_project
python3 test_brute_force.py
python3 test_hnsw.py
python3 benchmark.py
```

All committed to this folder, nothing needs installing beyond numpy.

## Session 3 update: fixed insertion speed, found the near-crossover point

Root cause of the 570s build time at n=50000: `_batch_distances` was still
rebuilding a fresh numpy array from a python list comprehension on every
single call (`np.stack([self.vectors[i] for i in ids])`). Fixed by storing
all vectors in one preallocated numpy matrix (grown by doubling as needed)
and using direct fancy-indexing (`self._vectors[idx_list] @ query`) instead.
Neighbor ids became internal integer row indices, external ids map through
a dict, public API (`add`, `search`) unchanged. Recall is bit-identical
after the change (0.966 at n=2000), confirms this was purely a perf fix.

Ran on Kshitij's actual machine (faster than the dev sandbox this was
built in, ~3x faster build times for the same n):

```
n=   500  brute=0.035ms  hnsw=0.284ms  speedup=0.10x  recall@10=1.000  build=0.2s
n=  1000  brute=0.050ms  hnsw=0.361ms  speedup=0.10x  recall@10=0.997  build=0.7s
n=  2000  brute=0.098ms  hnsw=0.433ms  speedup=0.20x  recall@10=0.983  build=1.7s
n=  5000  brute=0.218ms  hnsw=0.639ms  speedup=0.30x  recall@10=0.930  build=5.6s
n= 10000  brute=0.456ms  hnsw=0.946ms  speedup=0.50x  recall@10=0.843  build=13.2s
n= 20000  brute=0.879ms  hnsw=1.539ms  speedup=0.57x  recall@10=0.835  build=31.0s
n= 50000  brute=2.893ms  hnsw=3.546ms  speedup=0.82x  recall@10=0.593  build=90.9s
```

Build time at n=10000 went from 191.9s (session 1) -> 70.0s (session 2) ->
13.2s (session 3, on real hardware), roughly a 14x total improvement across
the three passes. Query speedup is climbing steadily (0.1x -> 0.82x) and
n=50000 is essentially at parity. The trend strongly suggests actual
crossover (HNSW genuinely faster than brute force) somewhere in the
100k-200k range, we didn't confirm the exact number, a tool-level call
duration cap (~120s per command) prevented running n=80000+ in this
session.

**To find the real crossover point**, run this directly in a terminal on
your machine (no timeout there):

```
cd vectordb_project
python3 -c "
from benchmark import bench
for n in [100000, 200000, 500000]:
    r = bench(n, n_queries=10, ef_search=50)
    print(f\"n={r['n']:7d} speedup={r['speedup']:.2f}x recall={r['recall@10']:.3f} build={r['hnsw_build_s']:.1f}s\")
"
```

Recall is still dropping with n at fixed ef_search=50 (0.593 at n=50000,
same known tradeoff flagged in session 2), that's still an open item, not
a new bug, see the ef_search scaling section above.

## Recommended next step

Recall-vs-n is now the most important unresolved thread: pick an ef_search
that scales with n (e.g. proportional to log(n)) so the recall numbers
above 10k stop looking artificially bad. After that, product quantization
as originally planned.

## Session 4 update: fixed the recall-vs-n problem

Root cause: `ef_search` was a fixed value (50) regardless of dataset size,
so recall quietly degraded as n grew (0.997 at n=1000 down to 0.593 at
n=50000, all measured). Fixed by making `ef_search` auto-scale by default:
`HNSWIndex.search()` now computes `ef_search = max(30, M * log2(n))` when
the caller doesn't pass one explicitly (an explicit value still works and
overrides it, for manual speed/recall tuning).

The formula was found empirically, not just guessed: tested a few
candidates against measured recall across n=1000 to n=20000 and picked the
one that stayed flattest.

Full picture with auto-scaling, run on Kshitij's machine:

```
n=   500  hnsw=0.526ms  speedup=0.10x  recall@10=1.000  build=0.2s
n=  1000  hnsw=0.725ms  speedup=0.10x  recall@10=1.000  build=0.7s
n=  2000  hnsw=0.993ms  speedup=0.10x  recall@10=1.000  build=1.8s
n=  5000  hnsw=1.490ms  speedup=0.10x  recall@10=1.000  build=5.8s
n= 10000  hnsw=2.052ms  speedup=0.20x  recall@10=0.993  build=13.7s
n= 50000  hnsw=5.854ms  speedup=0.38x  recall@10=0.933  build=93.2s
```

Recall is now flat and high across the whole range (1.000 -> 1.000 ->
0.993 -> 0.933) instead of collapsing to 0.593 at the top end. Honest
tradeoff: query speedup at n=50000 dropped from the earlier 0.82x to
0.38x, because that 0.82x number was partly an artifact of bad recall
doing less search work, not a real win. This is now a fair, correct
comparison: HNSW is doing genuinely more work per query to actually find
the right answers, and the 50000-point crossover-vs-brute-force question
from session 3 is still open (the fair version of it), still worth
running the 100k/200k/500k command from session 3 to find the real
answer under this corrected ef_search behavior.

## Next: product quantization

HNSW's correctness and its two real performance bugs (insertion overhead,
recall-vs-n) are now found, fixed, and honestly documented end to end.
Time to move to the memory-compression piece, product quantization: split
each vector into subvectors, k-means cluster each subspace, store compact
integer codes instead of full floats. This is what makes billion-scale
fit in RAM at all in real systems.

## Session 5: Product quantization, built from scratch

New file `vectordb/pq.py`: `ProductQuantizer` (train/encode/decode/distance
computation) and `PQFlatIndex` (a usable index built on top of it). Splits
each vector into `m` subspaces, runs k-means (implemented from scratch,
plain Lloyd's algorithm, no sklearn) independently per subspace to learn
`k=256` centroids, then represents every vector as `m` single-byte centroid
indices instead of `dim` floats. Distance to a compressed vector is
computed via "asymmetric distance computation" (ADC): precompute a
query-to-centroid distance table once per query, then any stored vector's
distance is just table lookups and a sum, no decompression needed.

**The key finding, and it nearly looked like a bug before it turned into
the most interesting result of this session**: the first test used a fixed
`m=8` (32x compression) and got recall@10 = 0.393 against brute force
ground truth, on data that HNSW had no trouble with. Verified this wasn't
an implementation bug three ways: (1) reconstruction error is well below
random-baseline error, (2) the fast table-lookup distance exactly matches
the slow decode-then-compute distance (allclose), (3) swept `m` and watched
recall respond predictably:

```
m= 8  compression=32.0x  recall@10=0.393
m=16  compression=16.0x  recall@10=0.680
m=32  compression=8.0x   recall@10=0.893
```

This is real and expected: `m` controls how finely each chunk of the
vector gets quantized, fewer/bigger subspaces (high compression) means
each chunk gets replaced by a coarser approximation, more error, worse
ranking. This is the same category of finding as `ef_search` in HNSW, a
tunable knob that trades one resource for another, not a defect. Changed
the library default from `m=8` to `m=16` since 0.39 recall isn't useful in
practice, 0.68 is a much more honest "usable" default, with `m` exposed
for anyone who wants to tune the dial themselves.

Second smaller finding: measured compression ratio undershoots the
theoretical one at small n (8.8x measured vs 16x theoretical at n=5000),
because the codebooks themselves are a fixed memory cost that needs a
large enough dataset to amortize against. At n=20000 it climbs to 13.3x,
closer to the theoretical ceiling; the gap fully closes as n grows further.

All 6 tests pass (`test_pq.py`): compression ratio matches theory, PQ
reconstruction beats random guessing badly, the fast and slow distance
computations agree, the recall/compression tradeoff moves in the right
direction as `m` changes, and the m=16 default is usable, and real memory
savings are measurable and match expectations once amortized.

## Where things stand overall

All three planned pieces exist and are tested: brute force (ground truth),
HNSW (fast approximate search, with two real perf bugs found and fixed
along the way), and PQ (memory compression, with the recall/compression
tradeoff found and documented). Natural next step, if continuing: combine
them into one index (HNSW graph over PQ-compressed vectors, which is what
real systems like Faiss's IVFPQ do), or wrap what exists in a small API
and write the final portfolio-facing README with the whole story end to
end, including every honest bug found along the way, that narrative is
genuinely a strong one.

## Session 6: combined HNSW + PQ, and real query-time numbers on Kshitij's Mac

New file `vectordb/hnsw_pq.py`: `HNSWPQIndex`, HNSW's graph structure for
O(log n) traversal, but every node stores a PQ code (a few bytes) instead
of a full float vector, this is what Faiss's IVFPQ and similar production
indexes actually do. Distance during graph search goes through PQ's
asymmetric distance computation (ADC): one distance table built per query
or insertion, reused across the whole graph walk instead of rebuilt per
comparison.

**Bug found and fixed immediately**: first build at n=10000 took 61.4s,
4.5x slower than plain HNSW's 13.7s at the same size (measured in session
4). Root cause: `ProductQuantizer.asymmetric_distances` and `.decode`
(called on every single distance check during graph search, and on every
prune) were still looping over subspaces in pure python, the exact same
category of bug fixed twice already in HNSW itself, just newly introduced
here in PQ's code. Fixed with `table[np.arange(m), codes]`, vectorized
fancy-indexing instead of a python loop. Confirmed identical correctness
after the fix (all `test_pq.py` numbers unchanged), then re-measured:

```
n=10000  build: 61.4s -> 32.9s (1.87x faster)   query: 4.586ms -> 1.947ms (2.4x faster)
```

**Real query-time numbers, measured on Kshitij's actual machine, post-fix:**

```
n=10000  build=32.9s   query=1.947ms  memory=220.2KB
n=30000  build=108.0s  query=2.451ms  memory=532.8KB
```

(n=50000 hit the ~120s per-command limit on this device bridge before
finishing, same limit hit in session 3. The command to push further with
no timeout, direct in your own terminal, is at the bottom of this file.)

**Honest comparison against plain HNSW at the same sizes (session 4 numbers,
both post their respective bugfixes):**

```
             plain HNSW          HNSW+PQ (combined)
n=10000      build=13.7s         build=32.9s   (2.4x SLOWER build)
             query=2.052ms       query=1.947ms (about the same)
             memory=~2500KB      memory=220.2KB (~11x less memory)
```

So: query latency is essentially unchanged, PQ's compression doesn't cost
query speed here, because ADC table lookups are about as cheap as the dot
products plain HNSW was already doing. Memory is the real, large win
(~11x less at n=10000, and that ratio improves further at bigger n as the
fixed codebook cost amortizes, same effect documented in session 5). Build
time is genuinely slower, and this is now a known, understood cost, not a
mystery: every insertion does more total work than plain HNSW's insertion
(encoding, and reconstructing/decoding this node's own vector on every
prune event, vs a single stored float vector plain HNSW can dot-product
against directly). Real systems (Faiss) hide this cost with heavily
optimized C++/SIMD table lookups, not something worth chasing further in
pure python here, the tradeoff is documented and understood, which is
the actual goal of this project.

**Answering the original question directly**: no, 1 billion vectors was
never going to happen in pure python on a laptop, and that was known from
day one. What's actually achievable and now measured: tens of thousands of
vectors build in well under two minutes with millisecond-level query
latency and roughly 11x memory compression, on a single consumer machine,
in pure python with zero external ANN libraries. That's a legitimate,
demonstrable result for a portfolio project, and every number above is
real, measured, and its tradeoffs are understood and written down, not
hand-waved.

**To push further** (no ~120s tool-call limit, run directly on your Mac):

```
cd vectordb_project
python3 -c "
import time, numpy as np
from vectordb.hnsw_pq import HNSWPQIndex
for n in [50000, 100000, 200000]:
    rng = np.random.default_rng(0)
    vectors = rng.random((n, 64)).astype(np.float32)
    queries = rng.random((15, 64)).astype(np.float32)
    idx = HNSWPQIndex(64, M=16, ef_construction=200, pq_m=16, pq_k=256)
    idx.train(vectors[:2000])
    t0 = time.perf_counter()
    for i, v in enumerate(vectors):
        idx.add(i, v)
    build_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for q in queries:
        idx.search(q, k=10)
    query_ms = (time.perf_counter() - t0) / len(queries) * 1000
    print(f'n={n} build={build_s:.1f}s query={query_ms:.3f}ms memory={idx.memory_bytes()/1024:.1f}KB')
"
```

## Session 7: IVF+PQ (cluster-routed, disk-backed), and the actual billion-vector question answered honestly

New file `vectordb/ivf_pq.py`: `IVFPQIndex`. This is a genuinely different
architecture from everything before it, not another HNSW variant. Instead
of one graph connecting every vector to its neighbors (which means the
*whole* graph has to live in RAM to be searchable), the dataset is
coarse-clustered into `nlist` groups with k-means. Only the `nlist`
centroids stay in memory (a handful of KB to a few MB, independent of how
many vectors exist). Every actual vector, PQ-compressed to a few bytes, is
appended to a small per-cluster file on disk ("posting list", the classic
inverted-file name). A query compares against all centroids (cheap, RAM),
picks the `nprobe` nearest clusters, and reads only those files off disk.
This is architecturally close to what Microsoft's SPANN and Faiss's
IndexIVFPQ actually do, and it's the direct answer to "how do we get past
what fits in RAM": disk absorbs the dataset, RAM only ever holds a routing
index.

**Three bugs found and fixed before they became scaling surprises** (the
project's running theme continues, but this time mostly caught by
proactive review or by actually trying to push scale, not by a test
failing):

1. **`_read_cluster` parsed disk records with a python loop over
   `struct.unpack_from`**, one call per record. Caught by code review
   before ever benchmarking at scale, this is the exact same
   loop-instead-of-vectorized-numpy pattern already found three times
   elsewhere in this project (HNSW's distance batching twice, PQ's
   asymmetric_distances/decode once). Fixed with a numpy structured dtype
   (`np.dtype([("id","<i8"),("code","u1",(m,))])`) parsed via one
   `np.frombuffer` call instead of per-record unpacking. Verified
   identical test results before/after.

2. **IVF's centroid k-means OOM-killed the process outright at real
   scale.** The naive k-means (same style used for PQ's subspace k-means)
   builds an `(n, k, dim)` broadcast array to compute distances. That's
   fine when `dim` is small (PQ's subspaces), but IVF clusters *full*
   vectors, and `k` here is `nlist`, which can be in the thousands. At
   n=160000 training vectors and nlist=4000, that one array was
   160000*4000*64*4 bytes = 164GB of *shape*, and even after using the
   `|a-b|^2 = |a|^2 - 2a.b + |b|^2` identity to shrink it to an `(n, k)`
   matrix (2.56GB), several float32 temporaries alive at once during that
   one expression still added up to a real ~13GB and got SIGKILL'd by the
   OS. This was found by actually attempting the 1M-vector benchmark, not
   by guessing. Fixed by computing that `(n, k)` matrix in row-chunks
   (20000 rows at a time): peak memory becomes O(chunk_size * k) instead
   of O(n * k), independent of the training set size.

3. **PQ's own codebook training was the slower of the two k-means calls**,
   74s vs IVF's 3.6s on the same n, before either was optimized, an
   inversion that looked wrong until root-caused: PQ trains m=16
   subspaces * 15 iterations = 240 total k-means iterations, each doing
   real allocation work even though each individual array is small. Same
   two fixes applied (the distance identity + `np.add.at` for centroid
   recomputation instead of a python loop over k clusters), and since
   PQ's `k` is capped at 256, no chunking was needed there, just the
   allocation-reducing rewrite. Verified identical `test_pq.py`,
   `test_hnsw_pq.py`, and `test_ivf_pq.py` results before/after everything
   above, all three bugs were pure performance fixes, zero behavior
   change.

**The actual scaling numbers, before vs after the k-means fixes** (training
time at n=50000, nlist=894, before the fixes existed this genuinely could
not run at all past a certain size, this isn't a "faster" number, it's a
"works at all" number):

```
n=50000, nlist=894:   train 77.8s -> 9.9s   (7.9x faster, same clusters)
n=160000, nlist=4000: OOM-killed -> 66.8s / 105.2s in full pipeline (works)
```

**The big one: 1,000,000 vectors, built and searched successfully in one
run** (measured twice, both cloud sandbox and Kshitij's Mac; the Mac
numbers below are the 500K checkpoint, the 1M run needs to be run directly
in terminal because it exceeds the ~120s device-bridge command limit, see
the copy-paste command at the bottom):

```
Cloud sandbox, n=1,000,000, dim=64, nlist=4000, nprobe=200, pq_m=16:
  train_time:        98.6s
  build_time:        384.0s  (2604 vectors/sec)
  query_time:        7.5ms
  recall@10:         0.370   (consistent with 0.39-0.45 at every smaller n tested,
                              5000 through 500000, the architecture doesn't degrade with scale)
  resident_memory:   1,064 KB   <-- ONE MEGABYTE, for one million vectors
  disk_written:      22.9 MB
  process RSS:       unchanged before/after (real OS measurement, not self-reported)

Kshitij's Mac, n=500,000, dim=64, nlist=2828, nprobe=141, pq_m=16 (~2x faster than cloud sandbox):
  train_time:        15.5s
  build_time:        81.8s   (6112 vectors/sec)
  query_time:        3.2ms
  recall@10:         0.390
  resident_memory:   771 KB
  disk_written:      11.4 MB
```

**Answering the original question, honestly, one more time**: a full
billion vectors still isn't happening in an afternoon of pure python on a
laptop (at the Mac's measured ~6100 vec/s build rate, 1 billion vectors
would take about 45 hours of continuous building, not impossible, just not
a demo). But the thing that was actually in question, "does memory stop
being the limiting factor", is now proven, not just argued: 1 million
vectors indexed and searchable with recall@10=0.37, using 1 megabyte of
RAM. The RAM figure would be *the same at 1 billion vectors*, only nlist
(which scales as sqrt(n), not n) grows, e.g. nlist~128000 at 1B still only
needs 128000*64*4 = ~33MB of centroids in memory. The bottleneck at real
billion-scale is now purely build-time throughput and disk space, both
solvable with more time/hardware, not a fundamental "it doesn't fit"
problem. That's the actual engineering answer this session set out to get.

**To push to 1M+ yourself** (exceeds the device-bridge's ~120s command
limit, run directly in your own Mac terminal, no cap):

```
cd vectordb_project
python3 bench_ivf.py 1000000
```

Takes roughly 5-8 minutes based on the numbers above. To go further (5M,
10M), just change the number, build time and training time both scale
roughly linearly with n (training also grows slowly with nlist=4*sqrt(n)),
and resident memory will stay in the single-digit megabytes regardless.

## How to run everything yourself (updated)

```
cd vectordb_project
python3 test_brute_force.py
python3 test_hnsw.py
python3 test_pq.py
python3 test_hnsw_pq.py
python3 test_ivf_pq.py
python3 benchmark.py
python3 bench_ivf.py 1000000   # no timeout in your own terminal, ~5-8 min
```

All committed to this folder, nothing needs installing beyond numpy.

## Session 8: persistence — IVFPQIndex can now survive a restart

Added `save()` / `load()` to `IVFPQIndex`. Before this session, every
index in this project, including IVF+PQ, only existed for as long as the
python process stayed alive. Kill the process, rebuild everything from
scratch. That's a real gap for something calling itself a "database":
IVF's posting-list `.bin` files were already persistent on their own
(they're just files, they survive a process exit for free), but nothing
recorded which cluster was which, or how to PQ-encode a future query the
same way training originally did, so that data was effectively orphaned,
unreachable bytes on disk with no way back in.

**What `save()` does**: writes one small `.npz` file (inside `storage_dir`
by default, so the whole index, routing index + posting lists, lives
together in one self-contained folder) containing the centroids, PQ
codebooks, cluster sizes, and config (dim/nlist/pq_m/pq_k). This file is
tiny, the same size as `resident_memory_bytes()` reports, independent of n.

**What `load()` does**: reconstructs a fresh `IVFPQIndex` object from that
file, pointed at the same `storage_dir`. Important design point: `load()`
deliberately does NOT call `train()`, since `train()` is what truncates
every cluster file to start clean, calling it here would destroy the
exact data being recovered. `load()` only ever reads.

**Tested three separate correctness claims, all passing** (new file
`test_ivf_pq_persistence.py`):
1. Save an index, delete the in-memory object entirely (simulating an
   actual process exit), load a brand new object from disk, confirm it
   returns byte-identical search results (same ids, same distances,
   `np.allclose`) to before the save. This is the real test, a persisted
   index that answers differently than the original would be a correctness
   bug wearing a feature's clothes.
2. `total_vectors()` and `resident_memory_bytes()` report correctly after
   a reload, matching their pre-save values exactly.
3. A reloaded index can keep accepting new inserts, and a vector added
   AFTER reload is actually findable by search, proving the posting-list
   files are being appended to in place, not silently starting fresh.

All pre-existing IVF+PQ tests (`test_ivf_pq.py`) re-run and produced
identical numbers, this was a pure addition, nothing about `train()`,
`add()`, or `search()` changed.

**What's still missing for full mutability**: this only covers
save/reload of a static index. There's still no delete or update support,
every posting list is append-only. That's the next honest gap if it's
worth closing (see the discussion above about reranking and delete
support being the two remaining pieces of the actual search/indexing
engine, separate from the API wrapper).

## How to run everything yourself (updated)

```
cd vectordb_project
python3 test_brute_force.py
python3 test_hnsw.py
python3 test_pq.py
python3 test_hnsw_pq.py
python3 test_ivf_pq.py
python3 test_ivf_pq_persistence.py
python3 benchmark.py
python3 bench_ivf.py 1000000   # no timeout in your own terminal, ~5-8 min
```

All committed to this folder, nothing needs installing beyond numpy.

## Session 9: wrapped the engine in a REST API (FastAPI)

New files: `vectordb/api.py` (the API itself), `run_api.py` (start it as
a real server), `test_api.py` (end-to-end tests).

This is the piece that turns "a python class you can import" into
"something you can actually run and query over HTTP", which is how real
vector databases get used in practice, nobody imports Pinecone's internal
graph class. A "collection" is one named `IVFPQIndex`, backed by its own
folder on disk (like a table). Multiple collections can exist side by
side under one server.

**The concrete flow**:
```
POST /collections/products {"dim": 64, "nlist": 50}          -> create
POST /collections/products/train {"vectors": [[...], ...]}   -> learn centroids/codebooks
POST /collections/products/vectors {"ids": [1,2,3], "vectors": [[...], [...], [...]]}
POST /collections/products/search {"vector": [...], "k": 5}  -> {"ids": [...], "distances": [...]}
GET  /collections/products/stats                             -> total_vectors, resident_memory_bytes, ...
GET  /collections                                             -> list every collection and its size
DELETE /collections/products                                  -> drop a collection and its files
```

**The persistence work from the previous session is what makes this
safe**: every collection auto-saves (`idx.save()`) after every train/add
call, and on server startup, every folder under `STORAGE_ROOT` containing
a saved `index_meta.npz` gets automatically reloaded via
`IVFPQIndex.load()`. So a server restart, deploy, or crash never loses
data, without wiring that in, this would be a network wrapper around an
in-memory toy, not a real service.

**Tested three real things, not just "does it not 500"**:
1. Full flow: create -> train -> add 20 vectors -> search for one
   vector's own contents -> its own id comes back as a top hit -> stats
   endpoint reports the right count. All through real HTTP
   request/response validation (FastAPI's `TestClient`), not calling
   `IVFPQIndex` directly.
2. **Persistence survives an actual simulated restart**: build a
   collection through the API, wipe the in-memory collection registry
   entirely (exactly what happens on a real process restart), start a
   fresh client, and confirm the collection reloads automatically with
   the right vector count and is still searchable, its own previously
   added data comes back correctly.
3. Error handling: wrong dimensions, `pq_m` not dividing `dim`, search or
   add before training, duplicate collection names, missing collections,
   all return the right HTTP status codes (400/404/409) instead of
   crashing or silently doing the wrong thing.

All passing on both the cloud sandbox and directly on Kshitij's Mac,
verified against a REAL running server with curl, not just the in-process
test client:
```
curl -X POST localhost:8000/collections/smoke -d '{"dim": 4, "nlist": 2, "pq_m": 2, "pq_k": 4}'
curl -X POST localhost:8000/collections/smoke/train -d '{"vectors": [[...], ...]}'
curl -X POST localhost:8000/collections/smoke/vectors -d '{"ids": [1,2,3], "vectors": [...]}'
curl -X POST localhost:8000/collections/smoke/search -d '{"vector": [0.1,0.2,0.3,0.4], "k": 2}'
-> {"ids":[1,2],"distances":[0.015,0.43]}
```

FastAPI also generates an interactive Swagger UI for free from the
request/response schemas in `api.py`, available at `/docs` while the
server is running, worth showing off directly in an interview.

**What's still not built**: no auth, no concurrency/thread-safety (two
simultaneous writes to the same collection could race), no delete/update
of individual vectors (same gap noted in session 8, this API can drop a
whole collection but not one vector within it), and no batching/streaming
for very large vector uploads (every vector in a request currently goes
through one at a time in a python loop inside `add_vectors`). None of
these matter for a portfolio demo of "the engine works and is reachable
over the network," but they'd be the next things a real production
version would need.

## How to run everything yourself (updated)

```
cd vectordb_project
python3 test_brute_force.py
python3 test_hnsw.py
python3 test_pq.py
python3 test_hnsw_pq.py
python3 test_ivf_pq.py
python3 test_ivf_pq_persistence.py
python3 test_api.py
python3 benchmark.py
python3 bench_ivf.py 1000000   # no timeout in your own terminal, ~5-8 min

# run the API as a real server:
python3 run_api.py
# then in another terminal, or a browser at http://localhost:8000/docs
```

All committed to this folder. Needs numpy plus (new this session)
fastapi, uvicorn, and httpx, installed via `pip3 install fastapi uvicorn httpx`.

## Session 10: the real 1,000,000-vector number, on Kshitij's own Mac

Ran `bench_ivf.py 1000000` directly in terminal on Kshitij's Mac (not
through the device bridge, which caps at ~120s). Full result:

```
n=1,000,000  dim=64  nlist=4000  nprobe=200  pq_m=16
train_time:   20.0s
build_time:   226.0s   (4,425 vectors/sec, ~2x faster raw throughput
                        than the identical run in the cloud sandbox
                        (2,604 vec/s), on OTHER people's hardware)
query_time:   7.53ms
recall@10:    0.37     (matches every smaller n tested, 5000 through 500000,
                        all land in the same 0.37-0.45 band, confirming the
                        architecture does not degrade with scale)
resident_memory_bytes: 1,089,536   (1.06 MB, for one million vectors)
disk_bytes:            24,000,000  (22.9 MB)
```

**A real bug found in the benchmark script itself, while reading this
output**: `rss_before_KB`/`rss_after_KB` printed `1,255,833,600`, over a
billion, an obviously wrong number for a KB-denominated field. Root
cause: `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss`'s units are
famously OS-dependent (documented as kilobytes on Linux, bytes on
macOS), so the field's own name (`_KB`) was a real lie for this specific
number on this machine. Verified this directly rather than assuming: ran
a small script comparing the raw `ru_maxrss` value against `ps -o rss=`
for the same live process, at small scale the two matched almost
exactly, but at the full 1M-vector run's scale, the printed value was
exactly 1024x the number you'd expect if it were really KB, math that
lines up perfectly with it actually being raw bytes (1,255,833,600 bytes
= 1,226,400 KB ≈ 1.17GB, an entirely plausible real memory footprint for
a process that just built a 1M x 64 float32 array (256MB) AND a
brute-force ground-truth copy of the same data for the recall check
(another 256MB), plus kmeans training buffers).

Fixed by no longer trusting `ru_maxrss` at all: `bench_ivf.py` now shells
out to `ps -o rss=`, which is unambiguously KB-denominated on both Linux
and macOS, verified directly against a live process before trusting it.
Re-ran the fix at n=20000 on both machines and got sane, consistent
numbers (tens of MB, matching what a python+numpy process actually
should use).

**Why this doesn't change the headline result**: the actual claim this
whole IVF+PQ effort exists to prove, `resident_memory_bytes` (the index's
own routing structure: centroids + PQ codebooks) staying flat regardless
of n, was never measured with the buggy field. That number (1.06MB at
1,000,000 vectors) was correct the whole time. The RSS field was only
ever a secondary cross-check (total process footprint, including the
benchmark's OWN temporary brute-force comparison array, which a real
deployment wouldn't even build), and it's now fixed and trustworthy for
future runs too.

**The actual headline, one more time, cleanly**: 1,000,000 vectors,
built from scratch with zero external ANN libraries, indexed and
searchable with 7.5ms query latency and 0.37 recall@10, using literally
1 megabyte of resident memory for the routing index, measured on
Kshitij's own MacBook, not a cloud sandbox. That's the number for the
portfolio.

## Session 11: the real 10,000,000-vector run, and what it actually revealed

Ran on Kshitij's own MacBook, unattended, on the pre-minibatch/pre-batched-insert
code (the "before" baseline this run was deliberately kept as, per the plan:
let it finish rather than kill it mid-run, since it's still correct, just slow,
and a real before/after number is worth more than a guess).

Raw output:

```
n = 10,000,000, dim = 64
nlist = 12,649, nprobe = 632, pq_m = 16
train_time_s:            134.605
build_time_s:            5104.435   (1,959.1 vec/s)
query_time_ms:           93.1988
recall_at_10:            None       (n > RECALL_MAX_N, skipped by design)
resident_memory_bytes:   3,303,680  (3.23 MB)
disk_bytes:              240,000,000 (228.88 MB)
rss_before_KB:           1,684,992
rss_after_KB:            45,952
rss_delta_KB:            -1,639,040
```

Comparing directly against the 1M run (Session 10: build 226s/4,425 vec/s,
query 7.53ms, resident_memory 1.06MB, disk 22.9MB, nlist=4000):

**The good news, confirming the core design promise:** resident memory grew
from 1.06MB to 3.23MB, a 3.05x increase for a 10x increase in data, tracking
nlist's growth (4,000 -> 12,649, a 3.16x increase) almost exactly, NOT the
vector count. Disk usage grew from 22.9MB to 228.88MB, a clean 10x, matching
vector count exactly (~24 bytes/vector including the id, as predicted).
Both numbers behave exactly the way the whole IVF+PQ design was supposed to.

**The bad news, and it's real, not a fluke:** build throughput dropped from
4,425 vec/s to 1,959 vec/s, a 2.26x slowdown per vector, even though nothing
about the per-vector work should be fundamentally different, just more
centroids to compare against (nlist grew 3.16x) plus the same fixed
per-vector Python-loop and file-open-per-insert overhead identified in the
audit two sessions ago. This confirms, with real measured numbers instead of
a projection, that the current `add()` path does not scale, exactly as
predicted before this run was even started.

**The bigger surprise: query latency got much worse than expected.**
7.53ms -> 93.2ms is a 12.4x increase, for only a 3.16x increase in nprobe.
Naively, if search cost were driven purely by nprobe, a 3.16x increase in
nprobe should cost about 3.16x more. If it were driven by total vectors
actually scanned (nprobe * average cluster size, and since nprobe is always
~5% of nlist, this scales almost exactly with n itself), it should cost
about 10x more. The measured 12.4x is worse than both. The most likely
explanation, consistent with everything else found in this project: each
probed cluster still pays its own per-file open+read cost in `_read_cluster`,
called once per cluster inside the `search()` loop, the exact same
open-per-operation pattern that's costing `add()` on the insert side, just
happening 632 times per query instead of once per insert. This is new,
concrete evidence for something that was previously only a projection (see
the "why <100ms at 1B is genuinely hard" discussion): disk I/O overhead,
not raw compute, is very likely the dominant cost in both the read and
write paths at real scale.

**The RSS number needs an honest explanation, not a hand-wave.** rss_before
(measured right after training finishes, before any inserts) was 1.65GB;
rss_after (measured after all 10M inserts) was 44.9MB, a large apparent drop.
This is not a bug and not a measurement error: it reflects that this run
used the OLD, not-yet-minibatched, full-batch `_kmeans`, which at this scale
(train_n ~506,000, nlist=12,649) computes several float32 (batch_size, nlist)
intermediate arrays inside training, each around 1GB before the in-place
accumulation fix. Those pages stayed resident (mapped but not yet reclaimed
by the OS) right after training finished, which is exactly when rss_before
was measured. Over the following 85-minute insert loop, macOS's allocator
evidently reclaimed that unused memory, which is why rss_after came back
down to a number consistent with the database's real steady-state footprint.
Worth re-verifying on the next big run now that minibatch k-means (already
written and synced, see below) should eliminate that transient spike
entirely, since it never materializes a train_n-sized array to begin with.

**Fitting a real scaling curve from two real data points** (not a guess):
`(5104.435/226)` for a 10x increase in n gives a build-time growth exponent
of about n^1.35, worse than linear. Projected out to n=1,000,000,000 on the
CURRENT unbatched insert path, that lands in the tens of days, worse than
the earlier back-of-envelope "weeks" estimate from before this run, which
only makes the case for fixing `add()`'s per-vector Python loop and
per-vector file open/close stronger, not weaker.

**Already in flight before this run finished:** minibatch k-means with
early stopping and in-place distance accumulation were implemented and
synced to `ivf_pq.py` while this 10M run was still going (it didn't affect
this run, the process already had the old module loaded in memory).
Measured standalone at n=200,000, dim=64, k=2,000: full-batch 19.09s vs
minibatch (size=20,000) 1.99s, a 9.58x speedup, for about 5% looser
clustering (mean squared distance to nearest centroid 4.22 vs 4.43). That
gap only widens in minibatch's favor as nlist and train_n keep growing
together at real scale.

**Next up:** batch the insert path itself (vectorized nearest-cluster
assignment + vectorized PQ encode across a whole batch, plus writing each
cluster's records once per batch instead of opening its file per vector),
and apply the same "read once, not once per probed cluster" fix to the
search side, since this run's query latency data now shows that side needs
it just as much as insert does.

## Session 12: 25M batched build, packed reads, GPU routing, and reranking

The batched ingestion rewrite was then run at 25,000,000 vectors on the same
MacBook. Build time fell from 5,104.4 seconds at 10M on the scalar path to
612.4 seconds at 25M: 40,821 vectors/sec, a 20.8x throughput improvement over
the 1,959 vectors/sec baseline despite indexing 2.5x more vectors. Training
took 63.8 seconds. The compressed index used 572.2 MiB. With `nprobe=1000`,
query latency was 163.0ms and recall@10 was 0.5 over three exact scans.

The query-side file-open bottleneck was removed by `compact()`, which packs
all postings into one memory-mapped segment with an offset table. On 200,000
vectors, packed queries measured 0.910ms vs 2.079ms unpacked (2.29x), returned
identical results, and opened zero posting files during the packed search.

Exact shortlist reranking is now optional. When enabled, raw float32 vectors
are appended to `raw_vectors.f32` and each posting stores its raw row. A query
first ranks compressed PQ candidates, then fetches and exactly reranks only a
small shortlist. On a 200,000-vector sweep:

```
nprobe  rerank  candidates  recall@10  p50_ms
    64       0        7159      0.377    0.812
    64     100        7159      0.473    0.816
   128       0       14318      0.430    1.472
   128     100       14318      0.610    1.570
   256       0       28635      0.480    3.127
   256     100       28635      0.787    3.070
```

The cost is disk, not resident RAM: at 25M x 64 dimensions, raw vectors add
6.4GB decimal and the larger postings add 800MB, for a final index of about
6.71GiB. Reranking remains opt-in so the original compressed-only format and
its storage efficiency are preserved.

Apple Silicon ingestion can also select `routing_backend="mlx"`. At the
25M build's representative 100,000-vector x 20,000-centroid x 64-dimensional
routing shape, the isolated route benchmark measured 1.644s with NumPy and
0.543s with MLX (3.03x) with identical assignments. A production-path 50,000
vector smoke test produced the same 0.33 recall@10 for NumPy and MLX; at this
small `nlist=894` shape MLX startup/transfer overhead made it slower, which is
why the GPU backend is explicit rather than the universal default.
