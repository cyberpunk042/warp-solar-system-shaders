"""Semantic within-tolerance merge: fixed-distortion leader clustering, bounded per-row error, addressable ids."""
import numpy as np

import warp as wp

from warp_compress.semantic_merge import SemanticMergeStore, leader_merge

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _concept_kv(T=1500, d=48, C=80, sigma=0.06, switch=0.05, seed=0):
    # tokens are near-synonyms of a smaller concept set (the semantic-tier regime)
    rng = np.random.default_rng(seed)
    proto = rng.standard_normal((C, d)).astype(np.float32)
    c = np.empty(T, np.int64); c[0] = rng.integers(C)
    for t in range(1, T):
        c[t] = rng.integers(C) if rng.random() < switch else c[t - 1]
    return (proto[c] + sigma * rng.standard_normal((T, d))).astype(np.float32)


def test_per_row_bound_is_guaranteed():
    X = _concept_kv(seed=1)
    for tol in (0.3, 0.6, 0.9):
        leaders, assign = leader_merge(X, tol)
        err = np.sqrt(((X - leaders[assign]) ** 2).sum(1))
        assert err.max() <= tol + 1e-5                        # every row within tol of its leader (exact bound)


def test_store_bound_holds_up_to_fp16():
    X = _concept_kv(seed=2)
    st = SemanticMergeStore(X, tol=0.7, device=_DEV)
    assert st.max_row_error(X) <= 0.7 + 3e-3                  # +fp16 leader rounding


def test_fetch_matches_reconstruct():
    X = _concept_kv(seed=3)
    st = SemanticMergeStore(X, tol=0.6, device=_DEV)
    R = st.reconstruct()
    idx = np.random.default_rng(4).integers(0, st.n_rows, 300)
    assert np.allclose(st.fetch(idx), R[idx], atol=1e-6)      # random access == full decode


def test_higher_tol_merges_more_and_shrinks():
    X = _concept_kv(seed=5)
    lo = SemanticMergeStore(X, tol=0.3, device=_DEV)
    hi = SemanticMergeStore(X, tol=0.9, device=_DEV)
    assert hi.n_leaders < lo.n_leaders                        # more merging
    assert hi.ratio > lo.ratio
    assert hi.bits_per_value() < lo.bits_per_value()          # and it costs fewer bits


def test_tol_zero_is_near_lossless():
    X = _concept_kv(T=400, seed=6)
    st = SemanticMergeStore(X, tol=0.0, device=_DEV)
    # tol=0 merges only (near-)coincident rows; reconstruct is the fp16 input (leaders are the rows themselves)
    assert np.allclose(st.reconstruct(), X.astype(np.float16).astype(np.float32), atol=1e-6)


def test_ids_are_skewed_so_entropy_helps():
    # when merging is real, a few leaders absorb most rows -> the id stream is skewed -> RRR beats fixed width
    X = _concept_kv(T=2000, d=48, C=60, seed=7)
    st = SemanticMergeStore(X, tol=0.9, device=_DEV)
    fixed_width_bits = st.n_rows * st._bits                   # raw ceil(log2 L) bits per id
    assert st.wm.index_bytes() * 8 < fixed_width_bits         # entropy coder compresses the skewed ids


def test_attention_output_error_grows_gracefully():
    K = _concept_kv(T=1200, d=48, seed=8)
    V = _concept_kv(T=1200, d=48, seed=9)
    rng = np.random.default_rng(10)
    Q = rng.standard_normal((64, 48)).astype(np.float32)
    sc = 1.0 / np.sqrt(48)

    def _sm(z):
        z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)

    true = _sm((Q @ K.T) * sc) @ V
    errs = []
    for tol in (0.3, 0.6, 0.9):
        ks = SemanticMergeStore(K, tol=tol, device=_DEV); vs = SemanticMergeStore(V, tol=tol, device=_DEV)
        out = _sm((Q @ ks.reconstruct().T) * sc) @ vs.reconstruct()
        errs.append(float(np.mean((out - true) ** 2)))
    assert errs[0] <= errs[-1]                                # more merging -> more (bounded) output error
    assert errs[-1] < 1e-2                                    # still small at the aggressive setting
