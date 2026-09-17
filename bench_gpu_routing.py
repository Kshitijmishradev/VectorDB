"""Compare the existing NumPy/Accelerate IVF router with an MLX GPU router.

It benchmarks only coarse-centroid assignment, verifies the selected cluster
IDs, and uses the same isolated MLX backend as the production index.

Example:
    python bench_gpu_routing.py --vectors 100000 --nlist 20000
"""
import argparse
import json
import os
import time

import numpy as np

from vectordb.mlx_routing import MLXRouter

try:
    import mlx.core as mx
except ModuleNotFoundError as exc:
    raise SystemExit(
        "MLX is not installed. Install it in this venv with: pip install mlx"
    ) from exc
except ImportError as exc:
    raise SystemExit(f"MLX is installed but Metal GPU access is unavailable: {exc}") from exc


def route_numpy(vectors, centroids, batch_size=None):
    """Current production routing implementation, isolated for benchmarking."""
    nlist = len(centroids)
    if batch_size is None:
        target_bytes = 64 * 1024 * 1024
        batch_size = max(1, min(8192, target_bytes // (4 * nlist)))

    centroid_norms = (centroids ** 2).sum(axis=1)
    assignments = np.empty(len(vectors), dtype=np.int64)
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))
        chunk = vectors[start:end]
        distances = chunk @ centroids.T
        distances *= -2.0
        distances += (chunk ** 2).sum(axis=1)[:, None]
        distances += centroid_norms[None, :]
        assignments[start:end] = distances.argmin(axis=1)
    return assignments


def timed(callable_):
    start = time.perf_counter()
    value = callable_()
    return value, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark CPU vs Apple GPU IVF centroid routing")
    parser.add_argument("--vectors", type=int, default=100_000)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--nlist", type=int, default=20_000)
    parser.add_argument(
        "--gpu-chunks", default="512,1024,2048,4096",
        help="comma-separated GPU routing chunk sizes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", dest="json_output")
    args = parser.parse_args()

    if args.vectors <= 0 or args.dim <= 0 or args.nlist <= 0:
        parser.error("vectors, dim, and nlist must be positive")
    gpu_chunks = [int(value) for value in args.gpu_chunks.split(",")]
    if not gpu_chunks or any(value <= 0 for value in gpu_chunks):
        parser.error("gpu chunk sizes must be positive")

    rng = np.random.default_rng(args.seed)
    vectors = rng.random((args.vectors, args.dim), dtype=np.float32)
    centroids = rng.random((args.nlist, args.dim), dtype=np.float32)

    cpu_chunk = max(1, min(8192, (64 * 1024 * 1024) // (4 * args.nlist)))
    print(f"shape: vectors={vectors.shape}, centroids={centroids.shape}")
    print(f"CPU chunk={cpu_chunk}")
    try:
        print(f"GPU: {mx.device_info(mx.gpu)}")
    except Exception:
        print("GPU: MLX Metal device")

    # Warm both backends so import, graph compilation, and first-dispatch costs
    # do not dominate the measured full routing pass.
    warm_n = min(1024, args.vectors)
    route_numpy(vectors[:warm_n], centroids, batch_size=cpu_chunk)
    gpu_router = MLXRouter(centroids)
    gpu_router.route(vectors[:warm_n], batch_size=min(1024, warm_n))

    cpu_ids, cpu_seconds = timed(
        lambda: route_numpy(vectors, centroids, batch_size=cpu_chunk))
    cpu_rate = args.vectors / cpu_seconds
    print(f"CPU  chunk={cpu_chunk:5d}  seconds={cpu_seconds:8.3f}  "
          f"vectors/s={cpu_rate:10.1f}")

    results = {
        "vectors": args.vectors,
        "dim": args.dim,
        "nlist": args.nlist,
        "cpu": {
            "chunk_size": cpu_chunk,
            "seconds": cpu_seconds,
            "vectors_per_second": cpu_rate,
        },
        "gpu": [],
    }

    for chunk_size in gpu_chunks:
        gpu_ids, gpu_seconds = timed(
            lambda size=chunk_size: gpu_router.route(vectors, batch_size=size))
        gpu_rate = args.vectors / gpu_seconds
        mismatches = int(np.count_nonzero(cpu_ids != gpu_ids))
        result = {
            "chunk_size": chunk_size,
            "seconds": gpu_seconds,
            "vectors_per_second": gpu_rate,
            "speedup_vs_cpu": cpu_seconds / gpu_seconds,
            "assignment_mismatches": mismatches,
            "assignment_match_percent": 100.0 * (args.vectors - mismatches) / args.vectors,
            "distance_matrix_MiB": chunk_size * args.nlist * 4 / (1024 * 1024),
        }
        results["gpu"].append(result)
        print(
            f"GPU  chunk={chunk_size:5d}  seconds={gpu_seconds:8.3f}  "
            f"vectors/s={gpu_rate:10.1f}  speedup={result['speedup_vs_cpu']:5.2f}x  "
            f"matches={result['assignment_match_percent']:.5f}%  "
            f"matrix={result['distance_matrix_MiB']:.1f}MiB")

    best = min(results["gpu"], key=lambda item: item["seconds"])
    print(f"best GPU chunk={best['chunk_size']}, "
          f"routing speedup={best['speedup_vs_cpu']:.2f}x")

    if args.json_output:
        json_dir = os.path.dirname(args.json_output)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as output_file:
            json.dump(results, output_file, indent=2)
            output_file.write("\n")


if __name__ == "__main__":
    main()
