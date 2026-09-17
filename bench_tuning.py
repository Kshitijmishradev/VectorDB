"""Sweep IVF nprobe and report the latency/recall tradeoff.

Run this on a representative in-memory sample before choosing the production
nprobe.  A percentage of nlist is not a quality target; recall is.
"""
import argparse
import os
import shutil
import tempfile
import time

import numpy as np

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_pq import IVFPQIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n", nargs="?", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=30)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--train-minibatch-size", type=int, default=0)
    parser.add_argument("--pq-train-size", type=int, default=0)
    parser.add_argument(
        "--routing-backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument(
        "--nprobes", default="",
        help="comma-separated probe counts; default uses an automatic sweep")
    parser.add_argument(
        "--rerank-values", default="0,50,100,200",
        help="comma-separated exact reranking shortlist sizes")
    args = parser.parse_args()

    rerank_values = sorted({int(value) for value in args.rerank_values.split(",")})
    if any(value < 0 for value in rerank_values):
        parser.error("rerank values must be non-negative")
    store_full_vectors = any(value > 0 for value in rerank_values)

    rng = np.random.default_rng(0)
    vectors = rng.random((args.n, args.dim), dtype=np.float32)
    queries = rng.random((args.queries, args.dim), dtype=np.float32)
    nlist = max(int(4 * np.sqrt(args.n)), 16)
    train_n = min(args.n, max(nlist * 40, 5000))
    storage = tempfile.mkdtemp(prefix="ivf_tuning_")

    try:
        index = IVFPQIndex(
            args.dim, nlist=nlist, pq_m=args.pq_m,
            storage_dir=storage, seed=0,
            store_full_vectors=store_full_vectors,
            routing_backend=args.routing_backend)
        index.train(
            vectors[:train_n],
            n_iters=args.train_iters,
            minibatch_size=args.train_minibatch_size or None,
            pq_train_size=args.pq_train_size or None,
        )
        build_start = time.perf_counter()
        index.add_batch(vectors, np.arange(args.n, dtype=np.int64))
        build_seconds = time.perf_counter() - build_start
        compact_start = time.perf_counter()
        index.compact()
        compact_seconds = time.perf_counter() - compact_start

        brute = BruteForceIndex(args.dim, metric="l2")
        brute.add(vectors)
        truth = [brute.search(query, k=10)[0] for query in queries]

        if args.nprobes:
            requested_probes = [int(value) for value in args.nprobes.split(",")]
        else:
            requested_probes = [
                4, 8, 16, 32, 64, 128, 256,
                nlist // 20, nlist // 10, nlist // 2, nlist,
            ]
        probes = sorted({min(nlist, probe) for probe in requested_probes if probe > 0})
        print(f"n={args.n} nlist={nlist} build={build_seconds:.3f}s "
              f"throughput={args.n / build_seconds:.0f} vec/s "
              f"compact={compact_seconds:.3f}s "
              f"raw_vectors={index.raw_vector_bytes() / (1024 * 1024):.1f}MiB "
              f"routing={args.routing_backend} "
              f"train_minibatch={args.train_minibatch_size or 'full'} "
              f"pq_train={args.pq_train_size or 'full'}")
        print("nprobe  rerank  candidates  recall@10  p50_ms  p95_ms")
        for nprobe in probes:
            for rerank in rerank_values:
                latencies = []
                hits = 0
                for query, true_ids in zip(queries, truth):
                    start = time.perf_counter()
                    got_ids, _ = index.search(
                        query, k=10, nprobe=nprobe, rerank=rerank)
                    latencies.append((time.perf_counter() - start) * 1000)
                    hits += len(set(true_ids.tolist()) & set(got_ids))
                print(f"{nprobe:6d}  {rerank:6d}  "
                      f"{args.n * nprobe / nlist:10.0f}  "
                      f"{hits / (10 * len(queries)):9.3f}  "
                      f"{np.percentile(latencies, 50):7.3f}  "
                      f"{np.percentile(latencies, 95):7.3f}")
    finally:
        if "index" in locals():
            index.close()
        shutil.rmtree(storage)


if __name__ == "__main__":
    main()
