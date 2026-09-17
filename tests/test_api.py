"""
End-to-end tests for the REST API, using FastAPI's TestClient (no real
network socket, but the full request/response/validation path really
runs). STORAGE_ROOT is pointed at an isolated temp folder BEFORE
importing vectordb.api, since api.py reads that env var at import time.
"""
import os
import shutil

TEST_STORAGE = "/tmp/vectordb_api_test_storage"
os.environ["VECTORDB_STORAGE_ROOT"] = TEST_STORAGE

import numpy as np
from fastapi.testclient import TestClient

from vectordb import api as api_module


def _clean():
    if os.path.exists(TEST_STORAGE):
        shutil.rmtree(TEST_STORAGE)
    api_module._collections.clear()


def test_full_flow_create_train_add_search():
    """The concrete end-to-end path described at the top of api.py: create
    a collection, train it, add vectors, search, and get a sane answer
    back, going through real HTTP request/response validation the whole
    way, not calling IVFPQIndex directly."""
    _clean()
    dim = 8
    np.random.seed(0)
    train_vectors = np.random.rand(50, dim).tolist()

    with TestClient(api_module.app) as client:
        r = client.post("/collections/demo", json={
            "dim": dim, "nlist": 5, "pq_m": 4, "pq_k": 8,
            "store_full_vectors": True,
            "pq_mode": "residual",
        })
        assert r.status_code == 200, r.text

        r = client.post("/collections/demo/train", json={
            "vectors": train_vectors, "coarse_training": "accumulated"})
        assert r.status_code == 200, r.text

        # add 20 known vectors
        vectors = np.random.rand(20, dim)
        r = client.post("/collections/demo/vectors", json={
            "ids": list(range(20)),
            "vectors": vectors.tolist(),
        })
        assert r.status_code == 200, r.text
        assert r.json()["total_vectors"] == 20

        r = client.post("/collections/demo/compact")
        assert r.status_code == 200, r.text
        assert r.json()["vectors"] == 20

        # search with the EXACT vector #7 as the query: it should come back as a top hit
        r = client.post("/collections/demo/search", json={
            "vector": vectors[7].tolist(), "k": 3, "nprobe": 5,
            "rerank": 10,
        })
        assert r.status_code == 200, r.text
        result = r.json()
        assert 7 in result["ids"], f"expected vector 7's own id to be a top match, got {result}"
        print(f"PASS: search for vector 7's own contents returned {result['ids']}, includes 7")

        r = client.get("/collections/demo/stats")
        stats = r.json()
        assert stats["total_vectors"] == 20
        assert stats["trained"] is True
        assert stats["compacted"] is True
        assert stats["store_full_vectors"] is True
        assert stats["routing_backend"] == "numpy"
        assert stats["pq_mode"] == "residual"
        assert stats["coarse_training"] == "accumulated"
        assert stats["raw_vector_bytes"] == 20 * dim * 4
        print(f"PASS: stats endpoint reports {stats}")

    _clean()


def test_persistence_survives_a_simulated_restart():
    """The point of wiring save()/load() into the API at all: build a
    collection, let the server 'restart' (fresh in-memory _collections
    dict, exactly what a real process restart looks like), and confirm
    the collection reloads automatically with all its data intact."""
    _clean()
    dim = 6
    np.random.seed(1)
    train_vectors = np.random.rand(30, dim).tolist()
    vectors = np.random.rand(10, dim)

    with TestClient(api_module.app) as client:
        client.post("/collections/persistent", json={
            "dim": dim, "nlist": 4, "pq_m": 2, "pq_k": 8,
            "store_full_vectors": True,
        })
        client.post("/collections/persistent/train", json={"vectors": train_vectors})
        client.post("/collections/persistent/vectors", json={
            "ids": list(range(10)), "vectors": vectors.tolist(),
        })
        r = client.post("/collections/persistent/compact")
        assert r.status_code == 200, r.text

    # simulate an actual process restart: wipe the in-memory registry,
    # nothing but what's on disk should determine what comes back
    api_module._collections.clear()
    assert "persistent" not in api_module._collections

    with TestClient(api_module.app) as client:
        r = client.get("/collections/persistent/stats")
        assert r.status_code == 200, r.text
        stats = r.json()
        assert stats["total_vectors"] == 10, f"expected reloaded collection to have 10 vectors, got {stats}"
        assert stats["trained"] is True
        assert stats["compacted"] is True

        r = client.post("/collections/persistent/search", json={
            "vector": vectors[3].tolist(), "k": 1, "nprobe": 4,
            "rerank": 10,
        })
        assert 3 in r.json()["ids"], "reloaded collection should still find its own data"
        print(f"PASS: collection survived simulated restart, stats={stats}, "
              f"search after reload found id 3: {r.json()}")

    _clean()


def test_error_handling():
    """Sanity checks on the guardrails: wrong dims, untrained search,
    duplicate creation, and missing collections should all fail loudly
    with sensible HTTP status codes, not silently do the wrong thing."""
    _clean()
    dim = 4

    with TestClient(api_module.app) as client:
        # missing collection
        r = client.get("/collections/ghost/stats")
        assert r.status_code == 404

        r = client.post("/collections/badcol", json={"dim": dim, "nlist": 2, "pq_m": 3})
        assert r.status_code == 400, "pq_m=3 doesn't divide dim=4, should be rejected"

        r = client.post("/collections/badgpu", json={
            "dim": dim, "nlist": 2, "pq_m": 2,
            "routing_backend": "cuda",
        })
        assert r.status_code == 400, "unsupported routing backend should be rejected"

        r = client.post("/collections/badpq", json={
            "dim": dim, "nlist": 2, "pq_m": 2, "pq_mode": "mystery"})
        assert r.status_code == 400

        client.post("/collections/errtest", json={
            "dim": dim, "nlist": 2, "pq_m": 2, "pq_k": 8})

        # duplicate creation
        r = client.post("/collections/errtest", json={
            "dim": dim, "nlist": 2, "pq_m": 2, "pq_k": 8})
        assert r.status_code == 409

        # search before training
        r = client.post("/collections/errtest/search", json={"vector": [0, 0, 0, 0], "k": 1})
        assert r.status_code == 400

        # add before training
        r = client.post("/collections/errtest/vectors", json={"ids": [1], "vectors": [[0, 0, 0, 0]]})
        assert r.status_code == 400

        # train with wrong dim
        r = client.post("/collections/errtest/train", json={"vectors": [[0, 0, 0]] * 10})
        assert r.status_code == 400

        # Finish training so routing-control validation reaches the search path.
        vectors = np.random.rand(20, dim).tolist()
        r = client.post("/collections/errtest/train", json={"vectors": vectors})
        assert r.status_code == 200
        client.post("/collections/errtest/vectors", json={
            "ids": list(range(20)), "vectors": vectors})
        r = client.post("/collections/errtest/search", json={
            "vector": vectors[0], "nprobe": 1, "max_candidates": 10})
        assert r.status_code == 400

        print("PASS: all error-handling checks behaved as expected")

    _clean()


if __name__ == "__main__":
    test_full_flow_create_train_add_search()
    test_persistence_survives_a_simulated_restart()
    test_error_handling()
    print("\nall API tests passed")
