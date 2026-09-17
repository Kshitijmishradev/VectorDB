"""
REST API wrapping the from-scratch vector DB engine (IVFPQIndex) built in
this project. This is the piece that turns "a python class you import"
into "something you can actually run as a service and query over HTTP",
which is how real vector databases (Pinecone, Weaviate, Qdrant) are
actually used, nobody imports Pinecone's internal graph class directly.

Concrete example of the whole flow, before the code:
  1. POST /collections/products {"dim": 64, "nlist": 50}
     -> creates a new, empty, untrained collection named "products"
  2. POST /collections/products/train {"vectors": [[...64 floats...], ...]}
     -> learns the IVF centroids + PQ codebooks from a sample
  3. POST /collections/products/vectors {"ids": [1,2,3], "vectors": [[...], [...], [...]]}
     -> compresses and writes those vectors to disk, routed to their nearest cluster
  4. POST /collections/products/search {"vector": [...64 floats...], "k": 5}
     -> returns the 5 closest ids and their approximate distances

Design: a "collection" is one named IVFPQIndex, backed by its own
storage_dir on disk (like a table in a normal database). Multiple
collections can exist side by side. Every mutating call (train, add)
immediately calls save() afterward, and on server startup, every
collection folder found under STORAGE_ROOT is automatically reloaded via
IVFPQIndex.load(). This means restarting the server (a deploy, a crash,
closing your laptop) never loses data, this is exactly what the
persistence work (save()/load()) from the previous session makes safe to
do. The per-request save() call is cheap: it only ever writes the small
routing index (centroids + codebooks + cluster sizes), independent of n,
never the actual vector data, which is already written straight to disk
by add() itself.
"""
import os
import shutil
from contextlib import asynccontextmanager
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .ivf_pq import IVFPQIndex

STORAGE_ROOT = os.environ.get("VECTORDB_STORAGE_ROOT", "./api_storage")

# name -> IVFPQIndex, the live in-memory registry of open collections.
# This dict IS the "server state", everything else just reads/writes it.
_collections: Dict[str, IVFPQIndex] = {}


def _collection_dir(name: str) -> str:
    return os.path.join(STORAGE_ROOT, name)


def _load_existing_collections():
    """Auto-reload every collection that was previously saved to disk, so
    restarting this process never loses data. Only folders that actually
    contain a saved index_meta.npz get reloaded, an untrained collection
    that was created but never trained+saved has nothing to recover,
    which is the correct behavior, there was nothing durable yet."""
    os.makedirs(STORAGE_ROOT, exist_ok=True)
    for name in os.listdir(STORAGE_ROOT):
        meta_path = os.path.join(_collection_dir(name), "index_meta.npz")
        if os.path.exists(meta_path):
            _collections[name] = IVFPQIndex.load(_collection_dir(name))
            print(f"[startup] reloaded collection '{name}' "
                  f"({_collections[name].total_vectors()} vectors)")


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _load_existing_collections()
    yield  # server runs here
    # nothing to do on shutdown: every mutation already saved as it happened


app = FastAPI(
    title="vectordb",
    description="A vector database built from scratch (brute force, HNSW, "
                 "product quantization, IVF), wrapped in a REST API.",
    lifespan=_lifespan,
)


def _get_collection(name: str) -> IVFPQIndex:
    if name not in _collections:
        raise HTTPException(status_code=404, detail=f"collection '{name}' not found")
    return _collections[name]


# ---------------------------------------------------------------- schemas

class CreateCollectionRequest(BaseModel):
    dim: int = Field(..., gt=0, description="vector dimensionality")
    nlist: int = Field(..., gt=0, description="number of coarse IVF clusters")
    pq_m: int = Field(16, gt=0, description="number of PQ subspaces, must divide dim")
    pq_k: int = Field(256, gt=0, description="centroids per PQ subspace")
    store_full_vectors: bool = Field(
        False, description="store float32 vectors to enable exact reranking")
    routing_backend: str = Field(
        "numpy", description="ingestion routing backend: numpy or mlx")


class TrainRequest(BaseModel):
    vectors: List[List[float]] = Field(
        ..., description="representative sample used to learn IVF centroids + PQ codebooks")


class AddRequest(BaseModel):
    ids: List[int]
    vectors: List[List[float]]


class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    nprobe: int = 8
    rerank: int = 0


class SearchResult(BaseModel):
    ids: List[int]
    distances: List[float]


class StatsResponse(BaseModel):
    total_vectors: int
    resident_memory_bytes: int
    nlist: int
    dim: int
    trained: bool
    compacted: bool
    store_full_vectors: bool
    routing_backend: str
    posting_bytes: int
    raw_vector_bytes: int


# --------------------------------------------------------------- endpoints

@app.post("/collections/{name}")
def create_collection(name: str, req: CreateCollectionRequest):
    if name in _collections:
        raise HTTPException(status_code=409, detail=f"collection '{name}' already exists")
    if req.dim % req.pq_m != 0:
        raise HTTPException(
            status_code=400,
            detail=f"dim ({req.dim}) must be divisible by pq_m ({req.pq_m})")
    if req.routing_backend not in {"numpy", "mlx"}:
        raise HTTPException(
            status_code=400,
            detail="routing_backend must be 'numpy' or 'mlx'")
    idx = IVFPQIndex(req.dim, req.nlist, pq_m=req.pq_m, pq_k=req.pq_k,
                      storage_dir=_collection_dir(name),
                      store_full_vectors=req.store_full_vectors,
                      routing_backend=req.routing_backend)
    _collections[name] = idx
    return {"status": "created", "name": name}


@app.post("/collections/{name}/train")
def train_collection(name: str, req: TrainRequest):
    idx = _get_collection(name)
    if idx.is_trained:
        raise HTTPException(status_code=409, detail=f"collection '{name}' is already trained")
    try:
        vectors = np.asarray(req.vectors, dtype=np.float32)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid vector batch: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[1] != idx.dim:
        raise HTTPException(
            status_code=400,
            detail=f"expected vectors of dim {idx.dim}, got shape {vectors.shape}")
    if len(vectors) < idx.nlist:
        raise HTTPException(
            status_code=400,
            detail=f"need at least nlist={idx.nlist} training vectors, got {len(vectors)}")
    idx.train(vectors)
    idx.save()
    return {"status": "trained", "training_vectors": len(vectors)}


@app.post("/collections/{name}/vectors")
def add_vectors(name: str, req: AddRequest):
    idx = _get_collection(name)
    if not idx.is_trained:
        raise HTTPException(status_code=400, detail="collection must be trained before adding vectors")
    if len(req.ids) != len(req.vectors):
        raise HTTPException(status_code=400, detail="ids and vectors must be the same length")
    if len(req.ids) == 0:
        raise HTTPException(status_code=400, detail="no vectors provided")

    try:
        vectors = np.asarray(req.vectors, dtype=np.float32)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"invalid vector batch: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[1] != idx.dim:
        raise HTTPException(
            status_code=400,
            detail=f"expected vectors with shape (n, {idx.dim}), got {vectors.shape}")
    idx.add_batch(vectors, np.asarray(req.ids, dtype=np.int64))

    idx.save()
    return {"status": "added", "count": len(req.ids), "total_vectors": idx.total_vectors()}


@app.post("/collections/{name}/search", response_model=SearchResult)
def search_collection(name: str, req: SearchRequest):
    idx = _get_collection(name)
    if not idx.is_trained:
        raise HTTPException(status_code=400, detail="collection must be trained before searching")
    query = np.asarray(req.vector, dtype=np.float32)
    if query.shape != (idx.dim,):
        raise HTTPException(
            status_code=400,
            detail=f"expected a flat vector of dim {idx.dim}, got shape {query.shape}")
    if req.k <= 0 or req.nprobe <= 0 or req.rerank < 0:
        raise HTTPException(
            status_code=400,
            detail="k and nprobe must be positive and rerank must be non-negative")
    if req.rerank and not idx.store_full_vectors:
        raise HTTPException(
            status_code=400,
            detail="reranking requires a collection created with store_full_vectors=true")

    ids, dists = idx.search(
        query, k=req.k, nprobe=req.nprobe, rerank=req.rerank)
    return SearchResult(ids=ids, distances=dists)


@app.post("/collections/{name}/compact")
def compact_collection(name: str):
    idx = _get_collection(name)
    if not idx.is_trained:
        raise HTTPException(status_code=400, detail="collection must be trained before compaction")
    result = idx.compact()
    return {
        "status": "compacted",
        "vectors": result["vectors"],
        "bytes": result["bytes"],
    }


@app.get("/collections/{name}/stats", response_model=StatsResponse)
def collection_stats(name: str):
    idx = _get_collection(name)
    return StatsResponse(
        total_vectors=idx.total_vectors(),
        resident_memory_bytes=idx.resident_memory_bytes() if idx.is_trained else 0,
        nlist=idx.nlist,
        dim=idx.dim,
        trained=idx.is_trained,
        compacted=idx.is_compacted,
        store_full_vectors=idx.store_full_vectors,
        routing_backend=idx.routing_backend,
        posting_bytes=idx.posting_bytes() if idx.is_trained else 0,
        raw_vector_bytes=idx.raw_vector_bytes() if idx.is_trained else 0,
    )


@app.get("/collections")
def list_collections():
    return {name: idx.total_vectors() for name, idx in _collections.items()}


@app.delete("/collections/{name}")
def delete_collection(name: str):
    idx = _get_collection(name)  # raises 404 if missing
    idx.close()
    del _collections[name]
    shutil.rmtree(_collection_dir(name), ignore_errors=True)
    return {"status": "deleted", "name": name}
