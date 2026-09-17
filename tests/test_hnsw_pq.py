import numpy as np

from vectordb.brute_force import BruteForceIndex
from vectordb.hnsw_pq import HNSWPQIndex


def _index(vectors, *, seed=7, batch=True, storage_dir=None,
           store_full_vectors=False):
    index = HNSWPQIndex(
        vectors.shape[1], M=8, ef_construction=60,
        pq_m=4, pq_k=16, seed=seed, initial_capacity=8,
        storage_dir=storage_dir, store_full_vectors=store_full_vectors)
    index.train(vectors, n_iters=5)
    ids = np.arange(len(vectors), dtype=np.int64)
    if batch:
        index.add_batch(vectors, ids)
    else:
        for vector_id, vector in zip(ids, vectors):
            index.add(int(vector_id), vector)
    return index


def test_tiny_sanity_check():
    rng = np.random.default_rng(0)
    vectors = rng.random((10, 32), dtype=np.float32)
    index = HNSWPQIndex(
        32, M=8, ef_construction=50, pq_m=8, pq_k=8, seed=0)
    index.train(vectors)
    index.add_batch(vectors, np.arange(len(vectors)))

    correct = sum(
        index.search(query, k=1, ef_search=20)[0][0] == vector_id
        for vector_id, query in enumerate(vectors))
    assert correct >= 7


def test_recall_and_memory_accounting_is_honest():
    rng = np.random.default_rng(1)
    vectors = rng.random((1000, 64), dtype=np.float32)
    queries = rng.random((20, 64), dtype=np.float32)

    brute = BruteForceIndex(64, metric="l2")
    brute.add(vectors)
    index = HNSWPQIndex(
        64, M=16, ef_construction=100, pq_m=16, pq_k=64, seed=1,
        initial_capacity=len(vectors))
    index.train(vectors, n_iters=5)
    index.add_batch(vectors, np.arange(len(vectors)))

    hits = 0
    for query in queries:
        truth, _ = brute.search(query, k=10)
        found, _ = index.search(query, k=10, ef_search=200)
        hits += len(set(truth.tolist()) & set(found))

    assert hits / (10 * len(queries)) >= 0.4
    assert brute.vectors.nbytes / index.code_bytes() > 3
    assert index.graph_bytes() > 0
    assert index.memory_bytes() >= index.code_bytes() + index.graph_bytes()


def test_batch_and_scalar_insertion_are_equivalent():
    rng = np.random.default_rng(2)
    vectors = rng.random((120, 16), dtype=np.float32)
    batch = _index(vectors, seed=11, batch=True)
    scalar = _index(vectors, seed=11, batch=False)

    np.testing.assert_array_equal(
        batch._codes[:len(batch)], scalar._codes[:len(scalar)])
    np.testing.assert_array_equal(
        batch._levels[:len(batch)], scalar._levels[:len(scalar)])
    np.testing.assert_array_equal(
        batch._neighbors0[:len(batch)], scalar._neighbors0[:len(scalar)])
    for query in vectors[:10]:
        batch_result = batch.search(query, k=5, ef_search=50)
        scalar_result = scalar.search(query, k=5, ef_search=50)
        assert batch_result[0] == scalar_result[0]
        np.testing.assert_allclose(batch_result[1], scalar_result[1])


def test_compact_graph_degree_limits_and_reciprocal_edges():
    rng = np.random.default_rng(3)
    vectors = rng.random((250, 16), dtype=np.float32)
    index = _index(vectors)

    assert np.all(index._counts0[:len(index)] <= index.M_max0)
    for layer, counts in index._upper_counts.items():
        assert np.all(counts[:index._upper_sizes[layer]] <= index.M)
    for node in range(len(index)):
        for layer in range(int(index._levels[node]) + 1):
            for neighbor in index._neighbors(node, layer):
                assert node in index._neighbors(int(neighbor), layer)


def test_symmetric_pruning_table_matches_decode_plus_adc():
    rng = np.random.default_rng(31)
    vectors = rng.random((80, 16), dtype=np.float32)
    index = _index(vectors)
    candidates = np.arange(1, 40, dtype=np.int64)
    reconstructed = index.pq.decode(index._codes[0:1])[0]
    adc_table = index.pq.distance_table(reconstructed)
    expected = index._batch_distances(candidates, adc_table)
    actual = index._symmetric_distances(0, candidates)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_exact_rerank_uses_mmap_sidecar(tmp_path):
    rng = np.random.default_rng(4)
    vectors = rng.random((300, 16), dtype=np.float32)
    index = _index(
        vectors, storage_dir=str(tmp_path), store_full_vectors=True)

    query = rng.random(16, dtype=np.float32)
    ids, distances = index.search(query, k=10, ef_search=100, rerank=50)
    expected = ((vectors[np.asarray(ids)] - query) ** 2).sum(axis=1)
    np.testing.assert_allclose(distances, expected, rtol=1e-6, atol=1e-6)
    assert distances == sorted(distances)
    assert index.raw_vector_bytes() == vectors.nbytes
    assert index.last_search_stats["candidates_returned"] == 50


def test_save_load_and_continued_insertion(tmp_path):
    rng = np.random.default_rng(5)
    vectors = rng.random((140, 16), dtype=np.float32)
    initial, extra = vectors[:100], vectors[100:]
    index = _index(
        initial, seed=19, storage_dir=str(tmp_path),
        store_full_vectors=True)
    before = index.search(vectors[120], k=7, ef_search=60, rerank=25)
    index.save()
    loaded = HNSWPQIndex.load(str(tmp_path))
    after = loaded.search(vectors[120], k=7, ef_search=60, rerank=25)
    assert before[0] == after[0]
    np.testing.assert_allclose(before[1], after[1])

    loaded.add_batch(extra, np.arange(100, 140, dtype=np.int64))
    assert len(loaded) == 140
    ids, distances = loaded.search(extra[0], k=5, ef_search=80, rerank=25)
    expected = ((vectors[np.asarray(ids)] - extra[0]) ** 2).sum(axis=1)
    np.testing.assert_allclose(distances, expected, rtol=1e-6, atol=1e-6)
    loaded.save()
    reloaded = HNSWPQIndex.load(str(tmp_path))
    assert reloaded.search(extra[0], k=5, ef_search=80, rerank=25)[0] == ids


def test_train_before_add_and_rerank_validation():
    index = HNSWPQIndex(dim=16, pq_m=4, pq_k=8)
    vector = np.random.default_rng(6).random(16, dtype=np.float32)
    try:
        index.add(0, vector)
        assert False, "expected add before train to fail"
    except AssertionError as error:
        assert "train" in str(error)

    training = np.random.default_rng(7).random((20, 16), dtype=np.float32)
    index.train(training)
    index.add_batch(training, np.arange(len(training)))
    try:
        index.search(vector, rerank=10)
        assert False, "expected rerank without raw vectors to fail"
    except ValueError as error:
        assert "store_full_vectors" in str(error)
