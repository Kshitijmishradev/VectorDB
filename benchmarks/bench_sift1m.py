"""Build and compare the from-scratch IVF+PQ and HNSW+PQ engines on SIFT1M."""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.datasets import ensure_sift1m, validate_sift_file
from vectordb.hnsw_pq import HNSWPQIndex
from vectordb.ivf_pq import IVFPQIndex


def _int_list(value):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return values


def _nonnegative_int_list(value):
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def _rss_kb():
    try:
        output = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True, text=True, check=False).stdout.strip()
        return int(output)
    except (OSError, ValueError):
        return -1


def _iter_rows(dataset, count, batch_size):
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        yield start, np.asarray(dataset[start:end], dtype=np.float32)


def _exact_subset_truth(base, count, queries, k=10, batch_size=50_000):
    """Exact top-k for a corpus prefix used by the HNSW tuning gate."""
    best_distances = np.full((len(queries), k), np.inf, dtype=np.float32)
    best_ids = np.full((len(queries), k), -1, dtype=np.int64)
    query_norms = (queries ** 2).sum(axis=1)
    for start, vectors in _iter_rows(base, count, batch_size):
        distances = vectors @ queries.T
        distances *= -2.0
        distances += (vectors ** 2).sum(axis=1)[:, None]
        distances += query_norms[None, :]
        vector_ids = np.arange(start, start + len(vectors), dtype=np.int64)
        for query_index in range(len(queries)):
            combined_distances = np.concatenate(
                (best_distances[query_index], distances[:, query_index]))
            combined_ids = np.concatenate((best_ids[query_index], vector_ids))
            selected = np.argpartition(combined_distances, k - 1)[:k]
            best_distances[query_index] = combined_distances[selected]
            best_ids[query_index] = combined_ids[selected]
    order = np.argsort(best_distances, axis=1)
    return np.take_along_axis(best_ids, order, axis=1)


def _save(rows, json_path, csv_path, chart_path):
    if not rows:
        return
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as output:
        json.dump(rows, output, indent=2)
        output.write("\n")
    fields = sorted({key for row in rows for key in row})
    with open(csv_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        config_dir = os.path.join(tempfile.gettempdir(), "vectordb_matplotlib")
        Path(config_dir).mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", config_dir)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axis = plt.subplots(figsize=(8, 5))
        colors = {"ivf": "#6f5cff", "hnsw": "#ef7d32"}
        for engine in ("ivf", "hnsw"):
            engine_rows = [row for row in rows if row["engine"] == engine]
            if not engine_rows:
                continue
            axis.plot(
                [row["p95_ms"] for row in engine_rows],
                [row["recall_at_10"] for row in engine_rows],
                marker="o", linestyle="none", color=colors[engine],
                label=engine.upper())
            for row in engine_rows:
                axis.annotate(
                    row["setting"], (row["p95_ms"], row["recall_at_10"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=7)
        axis.set_xlabel("p95 latency (ms)")
        axis.set_ylabel("recall@10")
        axis.set_title("SIFT recall-latency frontier")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(chart_path, dpi=160)
        plt.close(fig)
    except ImportError:
        print("matplotlib not installed; skipped chart", flush=True)


def _measure(name, engine, comparison, setting, queries, truth, search):
    first_start = time.perf_counter()
    search(queries[0])
    first_ms = (time.perf_counter() - first_start) * 1000

    latencies = []
    hits1 = hits10 = 0
    work = []
    for query, true_ids in zip(queries, truth):
        started = time.perf_counter()
        found, _, effort = search(query)
        latencies.append((time.perf_counter() - started) * 1000)
        hits1 += int(bool(found) and found[0] == int(true_ids[0]))
        hits10 += len(set(found) & set(true_ids.tolist()))
        work.append(effort)
    return {
        "name": name,
        "engine": engine,
        "comparison": comparison,
        "setting": setting,
        "queries": len(queries),
        "recall_at_1": round(hits1 / len(queries), 4),
        "recall_at_10": round(hits10 / (10 * len(queries)), 4),
        "first_query_ms": round(first_ms, 4),
        "p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies, 99)), 4),
        "qps": round(1000.0 / float(np.mean(latencies)), 3),
        "average_work": round(float(np.mean(work)), 1),
    }


def _build_ivf(base, n, dim, args, pq_mode, storage, controlled):
    if os.path.exists(storage):
        shutil.rmtree(storage)
    index = IVFPQIndex(
        dim, nlist=args.nlist, pq_m=args.pq_m, storage_dir=storage,
        seed=args.seed, store_full_vectors=max(args.rerank_values) > 0,
        routing_backend=args.routing_backend, pq_mode=pq_mode)
    # The controlled run gives both engines the identical first pq_train_size
    # vectors. The best IVF run retains the previously tuned larger coarse
    # sample and residual-specific random PQ sample.
    train_n = min(n, args.pq_train_size if controlled else args.train_size)
    training = np.asarray(base[:train_n], dtype=np.float32)
    started = time.perf_counter()
    index.train(
        training, n_iters=args.train_iters,
        minibatch_size=args.train_minibatch_size or None,
        pq_train_size=None if controlled else args.pq_train_size,
        coarse_training=args.coarse_training)
    train_seconds = time.perf_counter() - started
    del training

    started = time.perf_counter()
    for start, vectors in _iter_rows(base, n, args.batch_size):
        ids = np.arange(start, start + len(vectors), dtype=np.int64)
        index.add_batch(vectors, ids)
        end = start + len(vectors)
        if end == n or end % args.progress_every == 0:
            print(f"[ivf:{pq_mode}] indexed {end}/{n}", flush=True)
    build_seconds = time.perf_counter() - started
    started = time.perf_counter()
    index.compact()
    compact_seconds = time.perf_counter() - started
    return index, train_seconds, build_seconds, compact_seconds


def _benchmark_ivf(base, n, dim, queries, truth, args, comparison):
    rows = []
    modes = []
    if comparison in {"single", "controlled", "both"}:
        modes.append(("controlled", "standard", True))
    if comparison in {"best", "both"}:
        modes.append(("best", "residual", False))
    for label, pq_mode, controlled in modes:
        storage = os.path.join(args.storage_root, f"ivf_{label}")
        index, train_s, build_s, compact_s = _build_ivf(
            base, n, dim, args, pq_mode, storage, controlled)
        for _ in range(args.warmups):
            index.search(
                queries[_ % len(queries)], k=10,
                max_candidates=args.candidate_budgets[0],
                rerank=args.rerank_values[-1])
        for budget in args.candidate_budgets:
            for rerank in args.rerank_values:
                def search(query, budget=budget, rerank=rerank):
                    found, distances = index.search(
                        query, k=10, max_candidates=budget, rerank=rerank)
                    scanned = index._select_probe_clusters(
                        query, max_candidates=budget)[1]
                    return found, distances, scanned

                row = _measure(
                    f"ivf-{label}", "ivf", label,
                    f"budget={budget},rerank={rerank}",
                    queries, truth, search)
                row.update({
                    "n": n, "dim": dim, "pq_mode": pq_mode,
                    "pq_m": args.pq_m, "nlist": args.nlist,
                    "candidate_budget": budget, "rerank": rerank,
                    "ef_search": None, "M": None,
                    "train_time_s": round(train_s, 3),
                    "build_time_s": round(build_s, 3),
                    "compact_time_s": round(compact_s, 3),
                    "resident_index_bytes": index.resident_memory_bytes(),
                    "raw_vector_bytes": index.raw_vector_bytes(),
                    "index_bytes": index.disk_bytes(),
                    "process_rss_KB": _rss_kb(), "reused_index": False,
                })
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)
        index.close()
        if not args.keep_storage:
            shutil.rmtree(storage, ignore_errors=True)
    return rows


def _hnsw_storage(args, M):
    if args.hnsw_storage and len(args.M_values) == 1:
        return args.hnsw_storage
    root = args.hnsw_storage or args.storage_root
    return os.path.join(root, f"hnsw_M{M}")


def _build_or_load_hnsw(base, n, dim, args, M):
    storage = _hnsw_storage(args, M)
    metadata = os.path.join(storage, "hnsw_meta.npz")
    if args.reuse_index and os.path.exists(metadata):
        index = HNSWPQIndex.load(storage)
        expected = (dim, M, args.ef_construction, args.pq_m, n)
        actual = (index.dim, index.M, index.ef_construction, index.pq.m, len(index))
        if actual != expected:
            index.close()
            raise ValueError(
                f"persisted HNSW configuration {actual} does not match requested {expected}")
        return index, 0.0, 0.0, True

    shutil.rmtree(storage, ignore_errors=True)
    index = HNSWPQIndex(
        dim, M=M, ef_construction=args.ef_construction,
        pq_m=args.pq_m, seed=args.seed, initial_capacity=max(n, 1),
        storage_dir=storage, store_full_vectors=max(args.rerank_values) > 0)
    train_n = min(n, args.pq_train_size)
    training = np.asarray(base[:train_n], dtype=np.float32)
    started = time.perf_counter()
    index.train(training, n_iters=args.train_iters)
    train_seconds = time.perf_counter() - started
    del training

    started = time.perf_counter()
    for start, vectors in _iter_rows(base, n, args.batch_size):
        ids = np.arange(start, start + len(vectors), dtype=np.int64)
        index.add_batch(vectors, ids)
        end = start + len(vectors)
        if end == n or end % args.progress_every == 0:
            rate = end / max(time.perf_counter() - started, 1e-9)
            print(f"[hnsw:M={M}] indexed {end}/{n}, {rate:.0f} vec/s", flush=True)
    build_seconds = time.perf_counter() - started
    index.save()
    return index, train_seconds, build_seconds, False


def _benchmark_hnsw(base, n, dim, queries, truth, args, comparison):
    rows = []
    label = "controlled" if comparison in {"single", "controlled"} else comparison
    for M in args.M_values:
        index, train_s, build_s, reused = _build_or_load_hnsw(
            base, n, dim, args, M)
        fixed_bytes = index.pq.codebooks.nbytes + index._symmetric_tables.nbytes
        scalable_bytes = max(0, index.resident_memory_bytes() - fixed_bytes)
        projected = fixed_bytes + int(scalable_bytes * (1_000_000 / max(n, 1)))
        if n <= 100_000 and projected > args.memory_gate_gib * 1024 ** 3:
            index.close()
            raise RuntimeError(
                f"HNSW projected resident storage at 1M is {projected / 1024**3:.2f} GiB, "
                f"above the {args.memory_gate_gib:.1f} GiB gate")
        for _ in range(args.warmups):
            index.search(
                queries[_ % len(queries)], k=10,
                ef_search=args.ef_search_values[0],
                rerank=args.rerank_values[-1])
        for ef_search in args.ef_search_values:
            for rerank in args.rerank_values:
                def search(query, ef_search=ef_search, rerank=rerank):
                    found, distances = index.search(
                        query, k=10, ef_search=ef_search, rerank=rerank)
                    return found, distances, index.last_search_stats["nodes_visited"]

                row = _measure(
                    f"hnsw-M{M}", "hnsw", label,
                    f"M={M},ef={ef_search},rerank={rerank}",
                    queries, truth, search)
                row.update({
                    "n": n, "dim": dim, "pq_mode": "standard",
                    "pq_m": args.pq_m, "nlist": None,
                    "candidate_budget": None, "rerank": rerank,
                    "ef_search": ef_search, "M": M,
                    "ef_construction": args.ef_construction,
                    "train_time_s": round(train_s, 3),
                    "build_time_s": round(build_s, 3),
                    "compact_time_s": 0.0,
                    "resident_index_bytes": index.resident_memory_bytes(),
                    "projected_resident_bytes_at_1m": projected,
                    "raw_vector_bytes": index.raw_vector_bytes(),
                    "index_bytes": index.disk_bytes(),
                    "process_rss_KB": _rss_kb(), "reused_index": reused,
                })
                rows.append(row)
                print(json.dumps(row, indent=2), flush=True)
        index.close()
    return rows


def _mark_selected(rows):
    eligible = [
        row for row in rows
        if row["engine"] == "hnsw" and row["recall_at_10"] >= 0.95]
    selected = None
    if eligible:
        selected = min(
            eligible,
            key=lambda row: (
                row["p95_ms"], row["resident_index_bytes"],
                row["build_time_s"]))
    for row in rows:
        row["selected"] = row is selected
    return selected


def run(args):
    try:
        import h5py
    except ImportError as exc:
        raise SystemExit("install requirements-bench.txt to run SIFT1M") from exc

    dataset_path = ensure_sift1m(args.dataset, args.download)
    with h5py.File(dataset_path, "r") as handle:
        base, query_data, neighbor_data = validate_sift_file(handle)
        n = min(len(base), args.limit) if args.limit else len(base)
        dim = base.shape[1]
        query_count = len(query_data) if args.final else min(args.queries, len(query_data))
        queries = np.asarray(query_data[:query_count], dtype=np.float32)
        if n == len(base):
            truth = np.asarray(neighbor_data[:query_count, :10], dtype=np.int64)
        else:
            print(f"[ground truth] exact scan for {n} vectors", flush=True)
            truth = _exact_subset_truth(
                base, n, queries, k=10, batch_size=args.batch_size)
        rows = []
        if args.index in {"ivf", "both"}:
            rows.extend(_benchmark_ivf(
                base, n, dim, queries, truth, args, args.comparison))
        if args.index in {"hnsw", "both"}:
            rows.extend(_benchmark_hnsw(
                base, n, dim, queries, truth, args, args.comparison))

    selected = _mark_selected(rows)
    if selected:
        print("[selected HNSW] " + selected["setting"], flush=True)
    else:
        print("[selected HNSW] no configuration reached recall@10 >= 0.95", flush=True)
    _save(rows, args.json_output, args.csv_output, args.chart_output)
    return rows


def main():
    parser = argparse.ArgumentParser(description="SIFT IVF+PQ versus HNSW+PQ benchmark")
    parser.add_argument("--dataset", default="data/sift-128-euclidean.hdf5")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--index", choices=("ivf", "hnsw", "both"), default="ivf")
    parser.add_argument(
        "--comparison", choices=("single", "controlled", "best", "both"),
        default="single")
    parser.add_argument(
        "--preset", choices=("legacy", "optimized", "both"),
        help="backward-compatible IVF alias for --comparison")
    parser.add_argument("--final", action="store_true", help="use all 10,000 queries")
    parser.add_argument("--queries", type=int, default=200)
    parser.add_argument("--limit", type=int, help="100000 is the recommended HNSW gate")
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--candidate-budget", type=int, default=20_000)
    parser.add_argument("--candidate-budgets", type=_int_list)
    parser.add_argument("--pq-m", type=int, default=32)
    parser.add_argument("--rerank", type=int, default=100)
    parser.add_argument("--rerank-values", type=_nonnegative_int_list)
    parser.add_argument("--M", dest="M", type=int, default=16)
    parser.add_argument("--M-values", type=_int_list)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=200)
    parser.add_argument("--ef-search-values", type=_int_list)
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--hnsw-storage")
    parser.add_argument("--memory-gate-gib", type=float, default=14.0)
    parser.add_argument(
        "--coarse-training", choices=("legacy", "accumulated"), default="legacy")
    parser.add_argument("--routing-backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--train-minibatch-size", type=int, default=50_000)
    parser.add_argument("--pq-train-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--progress-every", type=int, default=100_000)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--storage-root", default="/tmp/vectordb_sift1m")
    parser.add_argument("--keep-storage", action="store_true")
    parser.add_argument("--json", dest="json_output", default="results/sift1m_compare.json")
    parser.add_argument("--csv", dest="csv_output", default="results/sift1m_compare.csv")
    parser.add_argument("--chart", dest="chart_output", default="results/sift1m_compare.png")
    args = parser.parse_args()
    args.candidate_budgets = args.candidate_budgets or [args.candidate_budget]
    args.rerank_values = args.rerank_values or [args.rerank]
    args.M_values = args.M_values or [args.M]
    args.ef_search_values = args.ef_search_values or [args.ef_search]
    if args.preset:
        args.index = "ivf"
        args.comparison = {
            "legacy": "controlled", "optimized": "best", "both": "both"
        }[args.preset]
        if args.preset == "legacy":
            args.pq_m = 16
            args.rerank_values = [0]
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.memory_gate_gib <= 0:
        parser.error("--memory-gate-gib must be positive")
    run(args)


if __name__ == "__main__":
    main()
