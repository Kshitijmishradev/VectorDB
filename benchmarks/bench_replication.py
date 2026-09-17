"""Benchmark-only 25% boundary replication gate on SIFT1M.

This intentionally does not alter the persistent index. It compares exact
ranking inside routed candidate sets at the same scanned-record budget; only a
gain of at least 0.02 recall@10 should justify implementing stored replicas.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.datasets import ensure_sift1m, iter_rows, validate_sift_file
from vectordb.ivf_pq import IVFPQIndex


def _two_nearest(vectors, centroids, batch_size=2048):
    first = np.empty(len(vectors), dtype=np.int32)
    second = np.empty(len(vectors), dtype=np.int32)
    margins = np.empty(len(vectors), dtype=np.float32)
    centroid_sq = (centroids ** 2).sum(axis=1)
    for start in range(0, len(vectors), batch_size):
        chunk = vectors[start:start + batch_size]
        dists = -2.0 * (chunk @ centroids.T)
        dists += (chunk ** 2).sum(axis=1)[:, None]
        dists += centroid_sq[None, :]
        pair = np.argpartition(dists, 1, axis=1)[:, :2]
        pair_dists = np.take_along_axis(dists, pair, axis=1)
        order = np.argsort(pair_dists, axis=1)
        pair = np.take_along_axis(pair, order, axis=1)
        pair_dists = np.take_along_axis(pair_dists, order, axis=1)
        end = start + len(chunk)
        first[start:end], second[start:end] = pair[:, 0], pair[:, 1]
        margins[start:end] = (
            (pair_dists[:, 1] - pair_dists[:, 0])
            / np.maximum(np.abs(pair_dists[:, 0]), 1e-6))
    return first, second, margins


def _group(ids, clusters, nlist):
    order = np.argsort(clusters, kind="stable")
    counts = np.bincount(clusters, minlength=nlist)
    offsets = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
    return ids[order], offsets, counts


def _candidate_ids(order, groups, offsets, counts, budget):
    pieces = []
    scanned = 0
    for cluster in order:
        start, end = offsets[cluster], offsets[cluster + 1]
        if end > start:
            pieces.append(groups[start:end])
            scanned += int(counts[cluster])
        if scanned >= budget:
            break
    return np.unique(np.concatenate(pieces)) if pieces else np.empty(0, np.int64), scanned


def run(args):
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("install requirements-bench.txt") from exc
    path = ensure_sift1m(args.dataset, args.download)
    with h5py.File(path, "r") as handle, tempfile.TemporaryDirectory() as storage:
        base, tests, neighbors = validate_sift_file(handle)
        train = np.asarray(base[:args.train_size], dtype=np.float32)
        index = IVFPQIndex(
            base.shape[1], args.nlist, pq_m=16, pq_k=256,
            storage_dir=storage, seed=args.seed)
        index.train(
            train, n_iters=args.train_iters,
            minibatch_size=args.train_minibatch_size)
        del train

        n = len(base)
        primary = np.empty(n, dtype=np.int32)
        secondary = np.empty(n, dtype=np.int32)
        margins = np.empty(n, dtype=np.float32)
        for start, rows in iter_rows(base, args.batch_size):
            first, second, margin = _two_nearest(
                np.asarray(rows, dtype=np.float32), index.centroids)
            end = start + len(rows)
            primary[start:end], secondary[start:end], margins[start:end] = (
                first, second, margin)
        replicate_n = int(n * args.replication_fraction)
        replicated_ids = np.argpartition(margins, replicate_n - 1)[:replicate_n]
        all_ids = np.arange(n, dtype=np.int64)
        base_groups, base_offsets, base_counts = _group(
            all_ids, primary, args.nlist)
        combined_ids = np.concatenate((all_ids, replicated_ids))
        combined_clusters = np.concatenate((primary, secondary[replicated_ids]))
        replica_groups, replica_offsets, replica_counts = _group(
            combined_ids, combined_clusters, args.nlist)

        base_hits = replica_hits = base_scanned = replica_scanned = 0
        query_count = min(args.queries, len(tests))
        for query_idx in range(query_count):
            query = np.asarray(tests[query_idx], dtype=np.float32)
            cluster_dists = index._centroid_norms - 2.0 * (index.centroids @ query)
            order = np.argsort(cluster_dists)
            base_ids, scanned = _candidate_ids(
                order, base_groups, base_offsets, base_counts, args.candidate_budget)
            replica_ids, replicated_scanned = _candidate_ids(
                order, replica_groups, replica_offsets, replica_counts,
                args.candidate_budget)
            truth = set(np.asarray(neighbors[query_idx, :10]).tolist())
            for candidate_ids, label in ((base_ids, "base"), (replica_ids, "replica")):
                candidate_vectors = np.asarray(base[candidate_ids], dtype=np.float32)
                dists = ((candidate_vectors - query) ** 2).sum(axis=1)
                found = candidate_ids[np.argsort(dists)[:10]]
                if label == "base":
                    base_hits += len(set(found.tolist()) & truth)
                else:
                    replica_hits += len(set(found.tolist()) & truth)
            base_scanned += scanned
            replica_scanned += replicated_scanned

    denominator = query_count * 10
    base_recall = base_hits / denominator
    replica_recall = replica_hits / denominator
    result = {
        "replication_fraction": args.replication_fraction,
        "candidate_budget": args.candidate_budget,
        "queries": denominator // 10,
        "base_recall_at_10": round(base_recall, 4),
        "replicated_recall_at_10": round(replica_recall, 4),
        "absolute_gain": round(replica_recall - base_recall, 4),
        "base_average_scanned_records": round(base_scanned / (denominator / 10), 1),
        "replicated_average_scanned_records": round(
            replica_scanned / (denominator / 10), 1),
        "passes_0_02_gate": bool(replica_recall - base_recall >= 0.02),
    }
    print(json.dumps(result, indent=2))
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description="SIFT boundary-replication gate")
    parser.add_argument("--dataset", default="data/sift-128-euclidean.hdf5")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--candidate-budget", type=int, default=20_000)
    parser.add_argument("--replication-fraction", type=float, default=0.25)
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--train-minibatch-size", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_output", default="results/sift_replication.json")
    args = parser.parse_args()
    if not 0 < args.replication_fraction <= 0.25:
        parser.error("--replication-fraction must be in (0, 0.25]")
    run(args)


if __name__ == "__main__":
    main()
