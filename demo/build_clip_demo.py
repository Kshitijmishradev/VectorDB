"""Build and evaluate a residual IVF+PQ index over CIFAR-100 CLIP vectors."""
import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demo.clip_tools import load_ml, normalize_rows, read_coarse_labels
from vectordb.ivf_pq import IVFPQIndex


def _embed_dataset(dataset, torch, model, preprocess, device, batch_size):
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            images = torch.stack([
                preprocess(dataset[index][0])
                for index in range(start, min(start + batch_size, len(dataset)))
            ]).to(device)
            features = model.encode_image(images).float()
            features /= features.norm(dim=-1, keepdim=True)
            embeddings.append(features.cpu().numpy().astype(np.float32))
            if start and start % 5000 == 0:
                print(f"embedded {start}/{len(dataset)}", flush=True)
    return np.concatenate(embeddings)


def build(args):
    artifact_dir = Path(args.artifact_dir)
    index_dir = artifact_dir / "index"
    if index_dir.exists():
        shutil.rmtree(index_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch, _, CIFAR100, model, preprocess, device = load_ml(args.data_root)
    train = CIFAR100(args.data_root, train=True, download=True)
    start = time.perf_counter()
    embeddings = _embed_dataset(
        train, torch, model, preprocess, device, args.embedding_batch_size)
    embedding_seconds = time.perf_counter() - start
    embeddings = normalize_rows(embeddings)
    np.save(artifact_dir / "train_embeddings.npy", embeddings)

    index = IVFPQIndex(
        embeddings.shape[1], args.nlist, pq_m=args.pq_m,
        storage_dir=str(index_dir), seed=args.seed,
        store_full_vectors=True, routing_backend=args.routing_backend,
        pq_mode="residual")
    train_start = time.perf_counter()
    index.train(
        embeddings, n_iters=args.train_iters,
        minibatch_size=args.train_minibatch_size,
        pq_train_size=args.pq_train_size,
        coarse_training=args.coarse_training)
    train_seconds = time.perf_counter() - train_start
    build_start = time.perf_counter()
    for offset in range(0, len(embeddings), args.index_batch_size):
        batch = embeddings[offset:offset + args.index_batch_size]
        index.add_batch(
            batch, np.arange(offset, offset + len(batch), dtype=np.int64))
    build_seconds = time.perf_counter() - build_start
    compact_start = time.perf_counter()
    index.compact()
    compact_seconds = time.perf_counter() - compact_start
    index.save()
    manifest = {
        "dataset": "CIFAR-100", "model": "OpenAI CLIP ViT-B/32",
        "vectors": len(embeddings), "dim": embeddings.shape[1],
        "nlist": args.nlist, "pq_m": args.pq_m,
        "pq_mode": "residual", "rerank": args.rerank,
        "candidate_budget": args.candidate_budget,
        "routing_backend": args.routing_backend, "device": device,
        "embedding_time_s": round(embedding_seconds, 3),
        "train_time_s": round(train_seconds, 3),
        "build_time_s": round(build_seconds, 3),
        "compact_time_s": round(compact_seconds, 3),
        "index_bytes": index.disk_bytes(),
        "class_names": train.classes,
    }
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    index.close()
    print(json.dumps(manifest, indent=2))


def evaluate(args):
    artifact_dir = Path(args.artifact_dir)
    torch, _, CIFAR100, model, preprocess, device = load_ml(args.data_root)
    train = CIFAR100(args.data_root, train=True, download=True)
    tests = CIFAR100(args.data_root, train=False, download=True)
    base_embeddings = np.load(artifact_dir / "train_embeddings.npy", mmap_mode="r")
    queries = _embed_dataset(
        [tests[index] for index in range(args.eval_queries)],
        torch, model, preprocess, device, args.embedding_batch_size)
    queries = normalize_rows(queries)
    index = IVFPQIndex.load(str(artifact_dir / "index"))
    manifest = json.loads((artifact_dir / "manifest.json").read_text())
    fine_labels = np.asarray(train.targets)
    test_fine = np.asarray(tests.targets[:args.eval_queries])
    coarse_labels = read_coarse_labels(args.data_root, "train")
    test_coarse = read_coarse_labels(args.data_root, "test")[:args.eval_queries]

    truth = []
    exact_batch = 25
    base = np.asarray(base_embeddings)
    for start in range(0, len(queries), exact_batch):
        scores = queries[start:start + exact_batch] @ base.T
        truth.extend(np.argsort(-scores, axis=1)[:, :10])

    recall_hits = fine_hits = coarse_hits = 0
    latencies = []
    for query_idx, (query, true_ids) in enumerate(zip(queries, truth)):
        start = time.perf_counter()
        found, _ = index.search(
            query, k=10, max_candidates=manifest["candidate_budget"],
            rerank=manifest["rerank"])
        latencies.append((time.perf_counter() - start) * 1000)
        recall_hits += len(set(found) & set(true_ids.tolist()))
        fine_hits += int((fine_labels[found] == test_fine[query_idx]).sum())
        coarse_hits += int((coarse_labels[found] == test_coarse[query_idx]).sum())
    denominator = len(queries) * 10
    result = {
        "queries": len(queries),
        "clip_bruteforce_recall_at_10": round(recall_hits / denominator, 4),
        "fine_label_precision_at_10": round(fine_hits / denominator, 4),
        "coarse_label_precision_at_10": round(coarse_hits / denominator, 4),
        "p50_ms": round(float(np.percentile(latencies, 50)), 4),
        "p95_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_ms": round(float(np.percentile(latencies, 99)), 4),
    }
    output_path = artifact_dir / "evaluation.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    index.close()
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description="CIFAR-100 CLIP index")
    parser.add_argument("command", choices=("build", "evaluate"))
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--artifact-dir", default="demo/artifacts")
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--pq-m", type=int, default=32)
    parser.add_argument("--candidate-budget", type=int, default=5000)
    parser.add_argument("--rerank", type=int, default=100)
    parser.add_argument("--routing-backend", choices=("numpy", "mlx"), default="numpy")
    parser.add_argument("--coarse-training", choices=("legacy", "accumulated"), default="legacy")
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--train-minibatch-size", type=int, default=20_000)
    parser.add_argument("--pq-train-size", type=int, default=50_000)
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--index-batch-size", type=int, default=10_000)
    parser.add_argument("--eval-queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    (build if args.command == "build" else evaluate)(args)


if __name__ == "__main__":
    main()
