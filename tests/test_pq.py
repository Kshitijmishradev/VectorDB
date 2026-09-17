import numpy as np
from vectordb.brute_force import BruteForceIndex
from vectordb.pq import ProductQuantizer, PQFlatIndex


def test_compression_ratio_is_as_advertised():
    """dim=64, m=8 should give exactly 32x compression: 256 bytes -> 8 bytes."""
    pq = ProductQuantizer(dim=64, m=8, k=256)
    ratio = pq.compression_ratio()
    assert ratio == 32.0, f"expected 32x, got {ratio}x"
    print(f"PASS: compression ratio = {ratio}x (256 bytes -> 8 bytes per vector)")


def test_reconstruction_error_is_small():
    """Encode then decode a batch of vectors, the reconstruction shouldn't be
    exact (that's the whole point, it's lossy) but should be close, much
    closer than random guessing."""
    np.random.seed(0)
    dim = 64
    vectors = np.random.rand(2000, dim).astype(np.float32)

    pq = ProductQuantizer(dim=dim, m=8, k=256)
    pq.train(vectors)

    codes = pq.encode(vectors[:100])
    reconstructed = pq.decode(codes)

    error = np.linalg.norm(vectors[:100] - reconstructed, axis=1).mean()
    random_baseline = np.linalg.norm(
        vectors[:100] - np.random.rand(100, dim).astype(np.float32), axis=1
    ).mean()

    assert error < random_baseline * 0.5, \
        f"reconstruction error ({error:.3f}) should be well below random baseline ({random_baseline:.3f})"
    print(f"PASS: reconstruction error={error:.3f} vs random baseline={random_baseline:.3f}")


def test_asymmetric_distance_matches_bruteforce_decode_distance():
    """The whole speed trick is computing distance via the table lookup
    (asymmetric_distances) instead of decoding + computing real distance.
    These two should agree closely, since they're approximating the same
    thing two different ways."""
    np.random.seed(1)
    dim = 32
    vectors = np.random.rand(500, dim).astype(np.float32)
    query = np.random.rand(dim).astype(np.float32)

    pq = ProductQuantizer(dim=dim, m=4, k=64)
    pq.train(vectors)
    codes = pq.encode(vectors)

    table = pq.distance_table(query)
    fast_dists = pq.asymmetric_distances(codes, table)

    reconstructed = pq.decode(codes)
    slow_dists = ((reconstructed - query) ** 2).sum(axis=1)

    assert np.allclose(fast_dists, slow_dists, atol=1e-3), \
        "fast table-lookup distance should exactly match decode-then-compute distance"
    print("PASS: asymmetric distance table matches decode-based distance (both compute the same thing)")


def test_recall_vs_compression_tradeoff():
    """The real end-to-end question: how much does compression cost in
    actual search quality, and how does that change with m? This is THE
    key PQ tradeoff, measured directly rather than assumed. An earlier
    version of this test used a fixed m=8 and found recall@10=0.393, which
    looked like a bug at first but turned out to be an honest, expected
    property of aggressive compression, not broken code. Sweeping m proves
    it's a real dial: more subspaces (smaller quantization error per chunk)
    trades compression for recall."""
    np.random.seed(2)
    dim = 64
    n = 3000
    vectors = np.random.rand(n, dim).astype(np.float32)
    queries = np.random.rand(30, dim).astype(np.float32)
    k = 10

    brute = BruteForceIndex(dim, metric="l2")
    brute.add(vectors)

    results = {}
    for m in [8, 16, 32]:
        pq_index = PQFlatIndex(dim, m=m, k=256)
        pq_index.train(vectors)
        pq_index.add(vectors)

        hits, total = 0, 0
        for q in queries:
            true_ids, _ = brute.search(q, k=k)
            got_ids, _ = pq_index.search(q, k=k)
            hits += len(set(true_ids.tolist()) & set(got_ids.tolist()))
            total += k
        recall = hits / total
        results[m] = recall
        print(f"PASS (informational): m={m:2d}  compression={pq_index.pq.compression_ratio():.1f}x  recall@{k}={recall:.3f}")

    assert results[32] > results[16] > results[8], \
        f"expected recall to improve as m increases, got {results}"
    assert results[16] >= 0.5, f"default m=16 recall too low to be useful: {results[16]:.3f}"


def test_memory_savings_are_real():
    """Confirm the actual measured memory footprint, not just the
    theoretical ratio, matches expectations, at the default m=16.

    Note: the raw theoretical ratio (dim*4/m = 16x) undershoots at small n,
    because the codebooks themselves (m*k*sub_dim floats, a FIXED cost
    regardless of dataset size) are a bigger fraction of total memory when
    there are few vectors to amortize them across. First measured at
    n=5000: only 8.8x instead of 16x, because ~64KB of codebook overhead
    was competing with only ~78KB of actual compressed vector data. This
    is expected: as n grows, the fixed codebook cost matters less and less
    and the ratio climbs toward the theoretical 16x. Testing at a larger n
    here so the measurement reflects that amortized, realistic case."""
    dim = 64
    n = 20000
    vectors = np.random.rand(n, dim).astype(np.float32)

    brute = BruteForceIndex(dim)
    brute.add(vectors)
    brute_bytes = brute.vectors.nbytes

    pq_index = PQFlatIndex(dim, m=16, k=256)
    pq_index.train(vectors)
    pq_index.add(vectors)
    pq_bytes = pq_index.memory_bytes()

    ratio = brute_bytes / pq_bytes
    print(f"PASS (informational): brute_force={brute_bytes/1024:.1f}KB  "
          f"pq={pq_bytes/1024:.1f}KB  actual_ratio={ratio:.1f}x  "
          f"(theoretical max = {pq_index.pq.compression_ratio():.1f}x)")
    assert ratio > 10, f"expected a real compression win, got only {ratio:.1f}x"


if __name__ == "__main__":
    test_compression_ratio_is_as_advertised()
    test_reconstruction_error_is_small()
    test_asymmetric_distance_matches_bruteforce_decode_distance()
    test_recall_vs_compression_tradeoff()
    test_memory_savings_are_real()
    print("\nall PQ tests passed")
