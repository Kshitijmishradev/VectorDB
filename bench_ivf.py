"""
Large-scale IVF+PQ benchmark: build time, query latency, memory
(self-reported + real process RSS), disk usage, and recall@10 against
brute force. Run with: python3 bench_ivf.py <n>

Designed to be run as standalone calls (one n per call) since each
device_bash call is capped at ~120s wall clock, unlike HNSW's O(n log n)
graph insertion, IVF's add() is just "nearest centroid + PQ encode +
append a few bytes to a file", so this should reach much larger n within
that same time budget, that's the whole point of building this.

IMPORTANT: this version STREAMS the test dataset in batches instead of
pre-generating one big (n, dim) array up front. The old version did
`vectors = rng.random((n, dim))` as its very first line, which is a real
scaling problem independent of how good IVFPQIndex's own memory story
is: at n=1,000,000, dim=64, that's 256MB just to hold the synthetic
INPUT data, at n=100,000,000 it's 25.6GB, at n=1,000,000,000 it's
256GB, impossible on a laptop no matter how small the actual index is.
Streaming in batches (see `batch_size`) keeps the benchmark's OWN memory
flat too, so it doesn't become the bottleneck before the database does.
"""
import argparse
import json
import sys
import os
import time
import shutil
import subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_pq import IVFPQIndex

STORAGE = "/tmp/ivf_bench_storage"

# Above this n, we skip the recall@10 check entirely. Not just for time,
# for memory too: computing recall requires an exact brute-force pass,
# which by definition needs the FULL dataset held in memory at once
# (that's what "exact" means). There's no way around that cost for a
# true ground-truth check, so this cap exists on purpose, it protects
# the recall step from becoming its own 25GB+ problem at huge n, the
# exact thing this whole rewrite is trying to avoid elsewhere.
RECALL_MAX_N = 2_000_000


def _rss_kb():
    """Real resident-set-size reading for this process, in KB, on both
    Linux and macOS. Deliberately NOT using python's own
    resource.getrusage().ru_maxrss: its units are documented as
    OS-dependent (KB on Linux, bytes on macOS per the classic docs), but
    when this was actually tested on Kshitij's Mac, a small manual check
    (raw value vs `ps`'s own KB reading for the same process) showed the
    raw ru_maxrss value matching KB almost exactly, yet a real 1M-vector
    benchmark run on the same machine printed a value 1024x too large to
    be KB and exactly consistent with being bytes instead, apparently
    inconsistent behavior for the same field on the same OS/python build.
    Rather than guess which one is right, this shells out to `ps -o rss=`,
    which is unambiguously KB-denominated on both Linux and macOS and was
    verified directly against a live process during this investigation."""
    pid = os.getpid()
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True).stdout.strip()
        return int(out)
    except (OSError, ValueError):
        # Sandboxed runners may deny process inspection.  Keep the actual
        # vector benchmark usable and mark RSS unavailable instead of failing.
        return -1


def _streaming_ground_truth(n, dim, queries, k, seed, batch_size,
                            progress_every=5_000_000):
    """Exact top-k without holding the full large dataset in memory."""
    rng = np.random.default_rng(seed + 1)
    query_sq = (queries ** 2).sum(axis=1)
    best_dists = np.full((len(queries), k), np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), k), -1, dtype=np.int64)
    processed = 0
    next_progress = progress_every

    while processed < n:
        this_batch = min(batch_size, n - processed)
        batch = rng.random((this_batch, dim), dtype=np.float32)
        dists = batch @ queries.T
        dists *= -2.0
        dists += (batch ** 2).sum(axis=1)[:, None]
        dists += query_sq[None, :]
        batch_ids = np.arange(processed, processed + this_batch, dtype=np.int64)

        for query_idx in range(len(queries)):
            candidate_dists = np.concatenate((best_dists[query_idx], dists[:, query_idx]))
            candidate_ids = np.concatenate((best_ids[query_idx], batch_ids))
            top = np.argpartition(candidate_dists, k - 1)[:k]
            best_dists[query_idx] = candidate_dists[top]
            best_ids[query_idx] = candidate_ids[top]

        processed += this_batch
        if processed >= next_progress or processed == n:
            print(f"  [exact recall scan: {processed}/{n}]", flush=True)
            next_progress += progress_every

    return best_ids


def bench(n, dim=64, n_queries=30, k=10, pq_m=16, nprobe=None, seed=0,
          train_iters=15, batch_size=100_000, progress_every=200_000,
          train_minibatch_size=None, pq_train_size=None,
          keep_storage=False, json_output=None, large_recall_queries=0,
          compare_unpacked_query=False, store_full_vectors=False, rerank=0,
          routing_backend="numpy"):
    if os.path.exists(STORAGE):
        shutil.rmtree(STORAGE)

    if rerank and not store_full_vectors:
        raise ValueError("rerank requires store_full_vectors=True")
    posting_record_bytes = 8 + pq_m + (8 if store_full_vectors else 0)
    expected_posting_bytes = n * posting_record_bytes
    expected_raw_bytes = n * dim * 4 if store_full_vectors else 0
    expected_disk_bytes = expected_posting_bytes + expected_raw_bytes
    free_disk_bytes = shutil.disk_usage(os.path.dirname(STORAGE)).free
    # Compaction writes the packed segment before deleting the source files,
    # so peak disk usage is roughly twice the final posting data.
    required_disk_bytes = int(
        expected_raw_bytes + 2.25 * expected_posting_bytes)
    if free_disk_bytes < required_disk_bytes:
        raise RuntimeError(
            f"insufficient free disk: need about {required_disk_bytes / 2**30:.2f} GiB "
            f"including headroom, have {free_disk_bytes / 2**30:.2f} GiB")

    nlist = max(int(4 * np.sqrt(n)), 16)
    if nprobe is None:
        nprobe = max(1, nlist // 20)  # probe ~5% of clusters

    train_n = min(n, max(nlist * 40, 5000))

    idx = IVFPQIndex(
        dim, nlist=nlist, pq_m=pq_m, pq_k=256,
        storage_dir=STORAGE, seed=seed,
        store_full_vectors=store_full_vectors,
        routing_backend=routing_backend)

    # --- train on a small representative sample. Bounded regardless of n
    # (see train_n above), so this was never the memory problem. ---
    train_rng = np.random.default_rng(seed)
    train_vectors = train_rng.random((train_n, dim), dtype=np.float32)
    t0 = time.perf_counter()
    idx.train(
        train_vectors,
        n_iters=train_iters,
        minibatch_size=train_minibatch_size,
        pq_train_size=pq_train_size,
    )
    train_time = time.perf_counter() - t0
    print(f"  [train done in {train_time:.1f}s, nlist={nlist} train_n={train_n}]", flush=True)
    del train_vectors

    # --- stream the full n vectors into the index in batches, never
    # holding more than one batch (batch_size rows) in memory at once.
    # This is the actual fix this rewrite exists for. ---
    data_rng = np.random.default_rng(seed + 1)  # a stream independent of the training sample
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    inserted = 0
    next_progress = progress_every
    while inserted < n:
        this_batch = min(batch_size, n - inserted)
        batch = data_rng.random((this_batch, dim), dtype=np.float32)
        batch_ids = np.arange(inserted, inserted + this_batch, dtype=np.int64)
        idx.add_batch(batch, batch_ids)
        inserted += this_batch
        del batch  # explicit: don't let this batch linger after we're done with it
        if inserted >= next_progress or inserted == n:
            elapsed = time.perf_counter() - t0
            print(f"  [build progress: {inserted}/{n} in {elapsed:.1f}s, {inserted/elapsed:.0f} vec/s]",
                  flush=True)
            next_progress += progress_every
    build_time = time.perf_counter() - t0

    query_rng = np.random.default_rng(seed + 2)
    queries = query_rng.random((n_queries, dim), dtype=np.float32)
    unpacked_query_time = None
    if compare_unpacked_query:
        t0 = time.perf_counter()
        for q in queries:
            idx.search(q, k=k, nprobe=nprobe, rerank=rerank)
        unpacked_query_time = (time.perf_counter() - t0) / n_queries
        print(f"  [unpacked query average: {unpacked_query_time * 1000:.3f} ms]",
              flush=True)

    compact_start = time.perf_counter()
    compact_result = idx.compact()
    compact_time = time.perf_counter() - compact_start
    print(f"  [compacted {compact_result['vectors']} vectors into one "
          f"{compact_result['bytes'] / (1024 * 1024):.1f} MiB segment in "
          f"{compact_time:.1f}s]", flush=True)
    rss_after = _rss_kb()

    t0 = time.perf_counter()
    for q in queries:
        idx.search(q, k=k, nprobe=nprobe, rerank=rerank)
    query_time = (time.perf_counter() - t0) / n_queries

    # --- recall@10 against brute force, only below RECALL_MAX_N (see comment above) ---
    recall = None
    recall_queries = 0
    recall_time = None
    if n <= RECALL_MAX_N:
        # regenerate the EXACT same sequence that was streamed into the index,
        # verified (see accompanying test) that a seeded numpy Generator drawn
        # in one big call reproduces bit-for-bit the same values as the same
        # seed drawn across several smaller sequential calls summing to the
        # same count, so this really is the same data the index saw, not a
        # different random sample that happens to look similar.
        verify_rng = np.random.default_rng(seed + 1)
        full_vectors = verify_rng.random((n, dim), dtype=np.float32)
        brute = BruteForceIndex(dim, metric="l2")
        brute.add(full_vectors)
        hits, total = 0, 0
        for q in queries:
            true_ids, _ = brute.search(q, k=k)
            got_ids, _ = idx.search(q, k=k, nprobe=nprobe, rerank=rerank)
            hits += len(set(true_ids.tolist()) & set(got_ids))
            total += k
        recall = hits / total
        recall_queries = len(queries)
        del full_vectors, brute
    elif large_recall_queries > 0:
        recall_queries = min(large_recall_queries, len(queries))
        recall_sample = queries[:recall_queries]
        recall_start = time.perf_counter()
        true_ids = _streaming_ground_truth(
            n, dim, recall_sample, k, seed, batch_size)
        hits = 0
        for query, query_true_ids in zip(recall_sample, true_ids):
            got_ids, _ = idx.search(
                query, k=k, nprobe=nprobe, rerank=rerank)
            hits += len(set(query_true_ids.tolist()) & set(got_ids))
        recall = hits / (recall_queries * k)
        recall_time = time.perf_counter() - recall_start

    posting_bytes = idx.posting_bytes()
    raw_vector_bytes = idx.raw_vector_bytes()
    disk_bytes = idx.disk_bytes()
    mem_bytes = idx.resident_memory_bytes()

    result = {
        "n": n, "dim": dim, "nlist": nlist, "nprobe": nprobe, "pq_m": pq_m,
        "store_full_vectors": store_full_vectors,
        "rerank": rerank,
        "routing_backend": routing_backend,
        "train_n": train_n,
        "train_minibatch_size": train_minibatch_size,
        "pq_train_size": pq_train_size,
        "train_time_s": round(train_time, 3),
        "build_time_s": round(build_time, 3),
        "compact_time_s": round(compact_time, 3),
        "total_index_time_s": round(train_time + build_time + compact_time, 3),
        "build_vectors_per_sec": round(n / build_time, 1),
        "query_time_ms": round(query_time * 1000, 4),
        "unpacked_query_time_ms": round(unpacked_query_time * 1000, 4)
        if unpacked_query_time is not None else None,
        "packed_query_speedup": round(unpacked_query_time / query_time, 3)
        if unpacked_query_time is not None else None,
        "recall_at_10": round(recall, 3) if recall is not None else None,
        "recall_queries": recall_queries,
        "recall_time_s": round(recall_time, 3) if recall_time is not None else None,
        "resident_memory_bytes": mem_bytes,
        "resident_memory_KB": round(mem_bytes / 1024, 2),
        "disk_bytes": disk_bytes,
        "disk_MB": round(disk_bytes / (1024 * 1024), 2),
        "posting_bytes": posting_bytes,
        "raw_vector_bytes": raw_vector_bytes,
        "rss_before_KB": rss_before,
        "rss_after_KB": rss_after,
        "rss_delta_KB": (rss_after - rss_before)
        if rss_before >= 0 and rss_after >= 0 else None,
    }
    for key, val in result.items():
        print(f"  {key}: {val}")

    if json_output:
        json_dir = os.path.dirname(json_output)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(json_output, "w", encoding="utf-8") as output_file:
            json.dump(result, output_file, indent=2)
            output_file.write("\n")

    if not keep_storage:
        idx.close()
        shutil.rmtree(STORAGE)
    else:
        print(f"  storage_kept_at: {STORAGE}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Streamed IVF+PQ scale benchmark")
    parser.add_argument("n", nargs="?", type=int, default=10_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--nprobe", type=int)
    parser.add_argument("--train-iters", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=100_000)
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--train-minibatch-size", type=int, default=0,
        help="coarse k-means sample per iteration; use 0 for exact full-batch training")
    parser.add_argument(
        "--pq-train-size", type=int, default=0,
        help="cap PQ codebook training sample; use 0 for the full training set")
    parser.add_argument("--keep-storage", action="store_true")
    parser.add_argument("--json", dest="json_output")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--compare-unpacked-query", action="store_true")
    parser.add_argument("--store-full-vectors", action="store_true")
    parser.add_argument(
        "--routing-backend", choices=("numpy", "mlx"), default="numpy",
        help="coarse ingestion routing backend; mlx uses Apple Silicon GPU")
    parser.add_argument(
        "--rerank", type=int, default=0,
        help="exactly rerank this many PQ finalists; enables raw-vector storage")
    parser.add_argument(
        "--recall-queries", type=int, default=3,
        help="exact streaming recall queries above 2M vectors; use 0 to skip")
    args = parser.parse_args()

    train_minibatch_size = args.train_minibatch_size or None
    pq_train_size = args.pq_train_size or None
    nlist = max(int(4 * np.sqrt(args.n)), 16)
    selected_nprobe = args.nprobe if args.nprobe is not None else max(1, nlist // 20)
    store_full_vectors = args.store_full_vectors or args.rerank > 0
    record_bytes = 8 + args.pq_m + (8 if store_full_vectors else 0)
    expected_mb = args.n * (
        record_bytes + (args.dim * 4 if store_full_vectors else 0)
    ) / (1024 * 1024)
    expected_candidates = args.n * selected_nprobe / nlist
    print(f"=== IVF+PQ benchmark: n={args.n} ===")
    print(f"  [preflight: nlist={nlist}, nprobe={selected_nprobe}, "
          f"routing={args.routing_backend}, rerank={args.rerank}, "
          f"expected_disk={expected_mb:.1f} MiB, "
          f"expected_candidates/query={expected_candidates:.0f}]", flush=True)
    if args.dry_run:
        raise SystemExit(0)
    bench(
        args.n,
        dim=args.dim,
        n_queries=args.queries,
        pq_m=args.pq_m,
        nprobe=args.nprobe,
        train_iters=args.train_iters,
        batch_size=args.batch_size,
        progress_every=args.progress_every,
        train_minibatch_size=train_minibatch_size,
        pq_train_size=pq_train_size,
        keep_storage=args.keep_storage,
        json_output=args.json_output,
        large_recall_queries=args.recall_queries,
        compare_unpacked_query=args.compare_unpacked_query,
        store_full_vectors=store_full_vectors,
        rerank=args.rerank,
        routing_backend=args.routing_backend,
    )
