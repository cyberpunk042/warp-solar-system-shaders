"""Randomized-Hadamard incoherence quant: orthogonal round-trip, row-addressable, measured outlier-error win."""
import numpy as np

import warp as wp

from warp_compress.hadamard_store import HadamardQuantStore, _fwht, _largest_pow2_divisor, _rht
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _outlier_weights(out=1024, inn=256, density=0.003, seed=0):
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((out, inn)) / np.sqrt(inn)).astype(np.float32)
    m = rng.random(W.shape) < density
    W[m] += rng.standard_normal(int(m.sum())).astype(np.float32) * 3.0        # a few big spikes
    return W


def test_fwht_is_self_inverse_up_to_scale():
    x = np.random.default_rng(1).standard_normal((5, 64)).astype(np.float32)
    assert np.allclose(_fwht(_fwht(x)), 64 * x, atol=1e-3)                    # FWHT^2 = b*I


def test_rht_is_orthogonal_roundtrip():
    W = _outlier_weights(256, 128, seed=2)
    signs = np.random.default_rng(0).choice(np.array([-1.0, 1.0], np.float32), size=128)
    b = _largest_pow2_divisor(128)
    back = _rht(_rht(W, signs, b, inverse=False), signs, b, inverse=True)
    assert np.allclose(back, W, atol=1e-4)                                   # rotate then un-rotate == identity


def test_reconstruct_is_lossless_over_the_rotated_quant():
    # the store is exactly: dequantize the rotated codes, then rotate back — reconstruct must equal that path
    W = _outlier_weights(seed=3)
    st = HadamardQuantStore(W, bits=4, device=_DEV)
    Wr_hat = st.store.reconstruct().reshape(W.shape)
    manual = _rht(Wr_hat, st.signs, st.block, inverse=True)
    assert np.allclose(st.reconstruct(), manual, atol=1e-6)


def test_rows_are_addressable():
    W = _outlier_weights(seed=4)
    st = HadamardQuantStore(W, bits=4, device=_DEV)
    R = st.reconstruct()
    idx = np.random.default_rng(5).integers(0, W.shape[0], 16)
    assert np.allclose(st.fetch_rows(idx), R[idx], atol=1e-6)                # per-row decode == full decode


def test_signs_regenerate_from_seed_zero_storage():
    W = _outlier_weights(64, 64, seed=6)
    a = HadamardQuantStore(W, bits=4, seed=7, device=_DEV)
    b = HadamardQuantStore(W, bits=4, seed=7, device=_DEV)
    assert np.array_equal(a.signs, b.signs)                                  # transform is a function of the seed
    assert a.bits_per_weight() == b.bits_per_weight()


def test_beats_direct_int4_on_outlier_weights():
    W = _outlier_weights(2048, 512, density=0.003, seed=8)
    x = np.random.default_rng(9).standard_normal((48, 512)).astype(np.float32)
    oerr = lambda R: float(np.mean((x @ R.T - x @ W.T) ** 2))
    direct = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV).reconstruct().reshape(W.shape)
    had = HadamardQuantStore(W, bits=4, device=_DEV)
    assert had.reconstruct().shape == W.shape
    assert oerr(had.reconstruct()) < oerr(direct)                           # incoherence lowers int4 output error
    assert had.bits_per_weight() < 8.0                                       # still an int4-class rate


def test_near_gaussian_is_an_honest_near_negative():
    # no outliers -> nothing to spread -> the rotation gives ~no benefit (the reported near-neutral case)
    Wg = (np.random.default_rng(10).standard_normal((1024, 256)) / 16).astype(np.float32)
    x = np.random.default_rng(11).standard_normal((32, 256)).astype(np.float32)
    oerr = lambda R: float(np.mean((x @ R.T - x @ Wg.T) ** 2))
    direct = QuantizedWeightStore(Wg, bits=4, huffman=True, device=_DEV).reconstruct().reshape(Wg.shape)
    had = HadamardQuantStore(Wg, bits=4, device=_DEV).reconstruct()
    assert oerr(had) < 1.5 * oerr(direct)                                   # within a hair either way (no big win)


def test_block_must_be_power_of_two_and_divide_in():
    W = _outlier_weights(32, 96, seed=12)                                    # 96 = 32*3
    HadamardQuantStore(W, bits=4, block=32, device=_DEV)                     # ok: 32 | 96
    for bad in (48, 5):                                                      # 48 not pow2; 5 doesn't divide 96
        try:
            HadamardQuantStore(W, bits=4, block=bad, device=_DEV)
            assert False, f"expected ValueError for block {bad}"
        except ValueError:
            pass
