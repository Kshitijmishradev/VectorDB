"""Optional Apple Silicon GPU backend for IVF coarse routing.

This module is deliberately separate from ``ivf_pq.py`` so importing and
using the database's default NumPy backend does not require MLX.  The router
keeps the centroid table on the Metal device and returns only cluster IDs to
the CPU; PQ encoding and persistence remain backend-independent.
"""
import numpy as np


class MLXRouter:
    """Reusable vector-to-centroid router backed by MLX/Metal."""

    def __init__(self, centroids: np.ndarray):
        try:
            import mlx.core as mx
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "routing_backend='mlx' requires the optional 'mlx' package"
            ) from exc
        except ImportError as exc:
            raise RuntimeError(f"MLX Metal GPU is unavailable: {exc}") from exc

        self.mx = mx
        self.centroids = mx.array(centroids)
        self.centroids_t = mx.transpose(self.centroids)
        self.centroid_norms = mx.sum(
            self.centroids * self.centroids, axis=1, stream=mx.gpu)
        mx.eval(self.centroids_t, self.centroid_norms)
        self.cache_bytes = int(centroids.nbytes + len(centroids) * 4)

    def route(self, vectors: np.ndarray, batch_size: int = 1024):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        mx = self.mx
        assignments = np.empty(len(vectors), dtype=np.int64)
        for start in range(0, len(vectors), batch_size):
            end = min(start + batch_size, len(vectors))
            chunk = mx.array(vectors[start:end])
            distances = mx.matmul(chunk, self.centroids_t, stream=mx.gpu)
            distances = mx.multiply(distances, -2.0, stream=mx.gpu)
            vector_norms = mx.sum(chunk * chunk, axis=1, stream=mx.gpu)
            distances = mx.add(distances, vector_norms[:, None], stream=mx.gpu)
            distances = mx.add(
                distances, self.centroid_norms[None, :], stream=mx.gpu)
            cluster_ids = mx.argmin(distances, axis=1, stream=mx.gpu)

            # MLX is lazy: evaluate before copying only the small ID array
            # back to NumPy, never the full distance matrix.
            mx.eval(cluster_ids)
            assignments[start:end] = np.asarray(cluster_ids, dtype=np.int64)
        return assignments
