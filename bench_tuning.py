"""Measure the IVF+PQ recall/latency frontier on deterministic synthetic data.

The quality columns isolate separate failure modes: ``routing_recall`` is the
fraction of exact neighbors whose IVF list was selected,
``pq_shortlist_recall`` is the fraction retained by PQ, and ``final_recall``
is recall after optional exact reranking.
"""
import argparse
import csv
import itertools
import json
import os
import shutil
import tempfile
import time

import numpy as np

from vectordb.brute_force import BruteForceIndex
from vectordb.ivf_pq import IVFPQIndex


def _ints(value):
    return [int(item) for item in value.split(",") if item]


def _strings(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def _select_result(results):
    passing = [row for row in results if row["passes_target"]]
    if not passing:
        return None
    return min(
        passing,
        key=lambda row: (row["index_bytes"], row["p95_ms"], row["build_time_s"]))


def _coarse_training_decision(results):
    grouped = {}
    for row in results:
        key = (row["nlist"], row["candidate_budget"], row["pq_mode"], row["pq_m"])
        grouped.setdefault(key, {}).setdefault(row["coarse_training"], {})[
            row["seed"]] = row
    comparisons = []
    for key, modes in grouped.items():
        shared = sorted(set(modes.get("legacy", {})) & set(modes.get("accumulated", {})))
        if len(shared) < 3:
            continue
        legacy = [modes["legacy"][seed] for seed in shared]
        accumulated = [modes["accumulated"][seed] for seed in shared]
        gain = float(np.mean([
            new["routing_recall_at_10"] - old["routing_recall_at_10"]
            for old, new in zip(legacy, accumulated)
        ]))
        comparisons.append({
            "nlist": key[0], "candidate_budget": key[1],
            "pq_mode": key[2], "pq_m": key[3], "seeds": shared,
            "mean_routing_recall_gain": round(gain, 4),
            "mean_legacy_train_time_s": round(float(np.mean([
                row["train_time_s"] for row in legacy])), 3),
            "mean_accumulated_train_time_s": round(float(np.mean([
                row["train_time_s"] for row in accumulated])), 3),
            "promote": bool(gain >= 0.005),
        })
    return comparisons


def _write_results(results, json_path=None, csv_path=None, acceptance=None):
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as output:
            json.dump({
                "acceptance": acceptance,
                "selected": _select_result(results),
                "coarse_training_comparisons": _coarse_training_decision(results),
                "frontier": results,
            }, output, indent=2)
            output.write("\n")
    if csv_path and results:
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)


def run(args):
    data_rng = np.random.default_rng(args.data_seed)
    vectors = data_rng.random((args.n, args.dim), dtype=np.float32)
    queries = data_rng.random((args.queries, args.dim), dtype=np.float32)
    brute = BruteForceIndex(args.dim, metric="l2")
    brute.add(vectors)
    truth = [brute.search(query, k=args.k)[0] for query in queries]

    nlists = (_ints(args.nlists) if args.nlists else [
        max(16, int(multiplier * np.sqrt(args.n))) for multiplier in (4, 8, 16)
    ])
    budgets = (_ints(args.candidate_budgets) if args.candidate_budgets else
               [max(args.k, int(args.n * ratio)) for ratio in (0.0125, 0.02, 0.025)])
    results = []

    configs = itertools.product(
        nlists, _ints(args.pq_ms), _strings(args.pq_modes),
        _strings(args.coarse_training), _ints(args.seeds))
    for nlist, pq_m, pq_mode, coarse_mode, seed in configs:
        if args.dim % pq_m:
            print(f"skip pq_m={pq_m}: does not divide dim={args.dim}")
            continue
        storage = tempfile.mkdtemp(prefix="ivf_frontier_")
        index = None
        try:
            train_n = min(args.n, max(nlist * 40, 5000))
            index = IVFPQIndex(
                args.dim, nlist=nlist, pq_m=pq_m,
                storage_dir=storage, seed=seed, store_full_vectors=True,
                routing_backend=args.routing_backend, pq_mode=pq_mode)
            train_start = time.perf_counter()
            index.train(
                vectors[:train_n], n_iters=args.train_iters,
                minibatch_size=args.train_minibatch_size or None,
                pq_train_size=args.pq_train_size or None,
                coarse_training=coarse_mode)
            train_seconds = time.perf_counter() - train_start
            build_start = time.perf_counter()
            index.add_batch(vectors, np.arange(args.n, dtype=np.int64))
            build_seconds = time.perf_counter() - build_start
            index.compact()

            # Assign only the exact top-k vectors. This computes the routing
            # ceiling without reading or materializing entire posting lists.
            truth_clusters = [index._nearest_clusters(vectors[ids]) for ids in truth]
            for budget, rerank in itertools.product(
                    budgets, _ints(args.rerank_values)):
                first_start = time.perf_counter()
                index.search(
                    queries[0], k=args.k, max_candidates=budget, rerank=rerank)
                first_ms = (time.perf_counter() - first_start) * 1000
                for warmup in range(args.warmups):
                    index.search(
                        queries[warmup % len(queries)], k=args.k,
                        max_candidates=budget, rerank=rerank)

                routing_hits = pq_hits = final_hits = 0
                latencies = []
                candidates = []
                shortlist_size = max(args.k, rerank)
                for query, true_ids, true_clusters in zip(
                        queries, truth, truth_clusters):
                    selected, scanned = index._select_probe_clusters(
                        query, max_candidates=budget)
                    routing_hits += int(np.isin(true_clusters, selected).sum())
                    shortlist_ids, _ = index.search(
                        query, k=shortlist_size, max_candidates=budget)
                    pq_hits += len(set(shortlist_ids) & set(true_ids.tolist()))
                    start = time.perf_counter()
                    found, _ = index.search(
                        query, k=args.k, max_candidates=budget, rerank=rerank)
                    latencies.append((time.perf_counter() - start) * 1000)
                    final_hits += len(set(found) & set(true_ids.tolist()))
                    candidates.append(scanned)

                denominator = len(queries) * args.k
                final_recall = final_hits / denominator
                p50 = float(np.percentile(latencies, 50))
                p95 = float(np.percentile(latencies, 95))
                row = {
                    "n": args.n, "dim": args.dim, "queries": len(queries),
                    "seed": seed, "nlist": nlist,
                    "candidate_budget": budget,
                    "average_candidates_scanned": round(float(np.mean(candidates)), 1),
                    "pq_mode": pq_mode, "pq_m": pq_m,
                    "coarse_training": coarse_mode, "rerank": rerank,
                    "routing_recall_at_10": round(routing_hits / denominator, 4),
                    "pq_shortlist_recall_at_10": round(pq_hits / denominator, 4),
                    "final_recall_at_10": round(final_recall, 4),
                    "first_query_ms": round(first_ms, 4),
                    "p50_ms": round(p50, 4),
                    "p95_ms": round(p95, 4),
                    "p99_ms": round(float(np.percentile(latencies, 99)), 4),
                    "qps": round(1000.0 / float(np.mean(latencies)), 3),
                    "train_time_s": round(train_seconds, 3),
                    "build_time_s": round(build_seconds, 3),
                    "index_bytes": index.disk_bytes(),
                    "resident_index_bytes": index.resident_memory_bytes(),
                    "passes_target": bool(
                        len(queries) >= 100 and final_recall >= args.target_recall
                        and p50 <= args.target_p50 and p95 <= args.target_p95),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
                acceptance = {
                    "recall_at_10_min": args.target_recall,
                    "p50_ms_max": args.target_p50,
                    "p95_ms_max": args.target_p95,
                    "minimum_queries": 100,
                }
                _write_results(
                    results, args.json_output, args.csv_output, acceptance)
        finally:
            if index is not None:
                index.close()
            shutil.rmtree(storage, ignore_errors=True)
    selected = _select_result(results)
    if selected is None:
        print("No configuration passed all acceptance targets; frontier retained.")
    else:
        print("Selected lowest-disk passing configuration:")
        print(json.dumps(selected, indent=2))
    return results


def main():
    parser = argparse.ArgumentParser(description="IVF+PQ recall/latency frontier")
    parser.add_argument("n", nargs="?", type=int, default=1_000_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--nlists", default="")
    parser.add_argument("--candidate-budgets", default="")
    parser.add_argument("--pq-modes", default="standard,residual")
    parser.add_argument("--pq-ms", default="16,32")
    parser.add_argument("--rerank-values", default="50,100,200")
    parser.add_argument("--coarse-training", default="legacy")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--data-seed", type=int, default=12345)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--train-minibatch-size", type=int, default=0)
    parser.add_argument("--pq-train-size", type=int, default=0)
    parser.add_argument("--routing-backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--target-p50", type=float, default=75.0)
    parser.add_argument("--target-p95", type=float, default=100.0)
    parser.add_argument("--json", dest="json_output")
    parser.add_argument("--csv", dest="csv_output")
    args = parser.parse_args()
    if args.queries < 1:
        parser.error("--queries must be positive")
    run(args)


if __name__ == "__main__":
    main()
