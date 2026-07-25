"""2:4 semi-structured sparsity codec: exact 2-of-4 pattern, addressable, measured honest-negative vs dense."""
import numpy as np

import warp as wp

from warp_compress.sparse_store import Sparse24WeightStore, _PAT_ARR
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _gauss(shape=(1024, 256), seed=0):
    return (np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def test_fetch_matches_reconstruct():
    W = _gauss(seed=1)
    st = Sparse24WeightStore(W, value_bits=4, device=_DEV)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(2).integers(0, st.n, 500)
    assert np.allclose(st.fetch(idx), R[idx], atol=1e-6)         # addressable == full decode


def test_pattern_keeps_exactly_two_lanes_per_group():
    # the STRUCTURAL 2:4 pattern is exactly 2 kept lanes per group (a kept value may quantize to 0, but the
    # slot is still structurally kept — what the hardware sees); the two NON-kept lanes are exactly zero
    W = _gauss(seed=3)
    st = Sparse24WeightStore(W, value_bits=8, device=_DEV)
    kept = _PAT_ARR[st.pattern]                                  # (n_groups, 2) kept lanes
    assert kept.shape[1] == 2 and np.all(kept[:, 0] < kept[:, 1])
    R = st.reconstruct().ravel().reshape(-1, 4)
    pruned = np.ones((st.n_groups, 4), bool)
    np.put_along_axis(pruned, kept, False, axis=1)               # mark kept lanes
    assert np.all(R[pruned] == 0.0)                              # every pruned lane is exactly zero


def test_keeps_the_two_largest_magnitude():
    W = _gauss((256, 128), seed=4)
    st = Sparse24WeightStore(W, value_bits=8, device=_DEV)
    g = W.ravel().reshape(-1, 4)
    kept_true = np.sort(np.argsort(-np.abs(g), axis=1)[:, :2], axis=1)
    kept_got = _PAT_ARR[st.pattern]                              # from the stored pattern, quant-independent
    assert np.array_equal(kept_got, kept_true)                  # the survivors are the two largest-|.|


def test_sparsity_is_one_half():
    st = Sparse24WeightStore(_gauss(seed=5), value_bits=8, device=_DEV)   # int8 -> kept values rarely hit 0
    assert st.sparsity == 0.5
    assert np.isclose(np.count_nonzero(st.reconstruct()) / st.n, 0.5, atol=0.02)


def test_honest_negative_worse_than_dense_int4():
    # the measured cost: post-training 2:4 (even at MORE bits) has higher output error than dense int4
    W = _gauss((2048, 512), seed=6)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((48, 512)).astype(np.float32)
    dense4 = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    sp8 = Sparse24WeightStore(W, value_bits=8, device=_DEV)
    oerr = lambda R: float(np.mean((x @ R.T - x @ W.T) ** 2))
    assert sp8.bits_per_weight() > dense4.bits_per_weight()      # 2:4-int8 even spends MORE bits
    assert oerr(sp8.reconstruct()) > oerr(dense4.reconstruct())  # yet is WORSE — the dropped 50% is gone


def test_pruned_positions_are_exact_zero():
    W = _gauss((128, 64), seed=8)
    st = Sparse24WeightStore(W, value_bits=4, device=_DEV)
    kept = _PAT_ARR[st.pattern]
    R = st.reconstruct().ravel().reshape(-1, 4)
    pruned = np.ones((st.n_groups, 4), bool)
    np.put_along_axis(pruned, kept, False, axis=1)
    assert np.all(R[pruned] == 0.0)                             # the two pruned lanes are exactly zero (≥2 zeros)


def test_size_must_be_multiple_of_four():
    W = np.zeros((3, 3), np.float32)                             # 9 not divisible by 4
    try:
        Sparse24WeightStore(W, value_bits=4, device=_DEV)
        assert False, "expected ValueError"
    except ValueError:
        pass
