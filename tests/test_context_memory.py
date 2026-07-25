"""Addressable compressed context memory: O(1) read + FM search + bounded semantic merge, over embeddings."""
import numpy as np

import warp as wp

from warp_compress.context_memory import ContextMemory

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _concept_context(N=2000, d=48, C=120, sigma=0.06, switch=0.06, seed=0):
    rng = np.random.default_rng(seed)
    proto = rng.standard_normal((C, d)).astype(np.float32)
    c = np.empty(N, np.int64); c[0] = rng.integers(C)
    for t in range(1, N):
        c[t] = rng.integers(C) if rng.random() < switch else c[t - 1]
    return (proto[c] + sigma * rng.standard_normal((N, d))).astype(np.float32), c


def test_at_is_addressable_and_matches_decode():
    X, _ = _concept_context(seed=1)
    mem = ContextMemory(X, tol=0.6)
    R = mem.reconstruct()
    for p in np.random.default_rng(2).integers(0, mem.n_tokens, 200):
        assert np.array_equal(mem.at(p), R[p])               # O(1) read == full decode


def test_reconstruction_within_tolerance():
    X, _ = _concept_context(seed=3)
    mem = ContextMemory(X, tol=0.7)
    assert mem.max_row_error(X) <= 0.7 + 3e-3                # semantic tier bound holds (+fp16)


def test_search_matches_brute_force():
    X, _ = _concept_context(seed=4)
    mem = ContextMemory(X, tol=0.7)
    probe = mem.ids[50:53]                                    # a real length-3 concept sub-sequence
    hits = mem.search(probe)
    brute = [i for i in range(mem.n_tokens - 2) if np.array_equal(mem.ids[i:i + 3], probe)]
    assert hits == brute and len(hits) >= 1                   # searchable in the compressed domain, exactly
    assert mem.count(probe) == len(brute)


def test_compresses_when_concepts_repeat():
    X, _ = _concept_context(N=4000, d=48, C=100, seed=5)
    mem = ContextMemory(X, tol=0.9)
    assert mem.V < mem.n_tokens                              # concepts merged
    assert mem.compression_x > 2.0                           # real compression vs raw fp16


def test_tol_zero_is_exact_book_and_addressable():
    X, _ = _concept_context(N=800, seed=6)
    mem = ContextMemory(X, tol=0.0)
    # exact content book: reconstruct == fp16 input; still searchable + addressable
    assert np.allclose(mem.reconstruct(), X.astype(np.float16).astype(np.float32), atol=1e-6)
    assert np.array_equal(mem.at(123), X[123].astype(np.float16).astype(np.float32))


def test_content_retrieval_recall():
    X, _ = _concept_context(N=3000, d=48, seed=7)
    tol = 0.7
    mem = ContextMemory(X, tol=tol)
    rng = np.random.default_rng(8)
    qpos = rng.integers(0, mem.n_tokens, 300)
    noisy = X[qpos] + 0.03 * rng.standard_normal((300, 48)).astype(np.float32)
    got = np.array([mem.nearest_id(q) for q in noisy])
    ret_err = np.sqrt(((mem.book[got] - X[qpos]) ** 2).sum(1))
    assert np.mean(ret_err <= tol + 0.2) > 0.9              # a noisy query retrieves a concept near the true token


def test_search_absent_pattern_is_empty():
    X, _ = _concept_context(N=1000, seed=9)
    mem = ContextMemory(X, tol=0.6)
    absent = [mem.V + 5, mem.V + 6]                          # ids that cannot occur
    assert mem.count(absent) == 0 and mem.search(absent) == []
