"""Lossless fp16/bf16 compression: entropy-coded exponent + raw sign/mantissa, exact bits, random access."""
import numpy as np

import warp as wp

from warp_compress.lossless_float import LosslessFloatStore, _pack_bits, _unpack_bits

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _fp16_bits(seed=0, n=40000):
    W = (np.random.default_rng(seed).standard_normal(n) / 8).astype(np.float32)
    return np.ascontiguousarray(W.astype(np.float16)).view(np.uint16).ravel()


def _bf16_bits(seed=0, n=40000):
    W = (np.random.default_rng(seed).standard_normal(n) / 8).astype(np.float32)
    return (np.ascontiguousarray(W).view(np.uint32).ravel() >> 16).astype(np.uint16)


def test_fp16_reconstruct_is_exact():
    u = _fp16_bits(1)
    st = LosslessFloatStore(u, 5, 10, device=_DEV)
    assert np.array_equal(st.reconstruct_u16(), u)                  # exact 16-bit patterns (lossless)


def test_bf16_reconstruct_is_exact():
    u = _bf16_bits(2)
    st = LosslessFloatStore(u, 8, 7, device=_DEV)
    assert np.array_equal(st.reconstruct_u16(), u)


def test_fetch_matches_random_positions():
    u = _bf16_bits(3)
    st = LosslessFloatStore(u, 8, 7, device=_DEV)
    idx = np.random.default_rng(4).integers(0, len(u), 1000)
    assert np.array_equal(st.fetch(idx), u[idx])


def test_bf16_compresses_below_16_bits():
    u = _bf16_bits(5)
    st = LosslessFloatStore(u, 8, 7, device=_DEV)
    assert st.bits_per_value() < 16.0                               # the low-entropy exponent shrinks


def test_recovered_floats_equal_the_originals():
    W = (np.random.default_rng(6).standard_normal((64, 64)) / 8).astype(np.float32)
    st = LosslessFloatStore.from_fp16(W, device=_DEV)
    recon = st.reconstruct_u16().view(np.float16).astype(np.float32).reshape(W.shape)
    assert np.array_equal(recon, W.astype(np.float16).astype(np.float32))   # exact fp16 values back


def test_bit_pack_unpack_roundtrips_odd_widths():
    for w in (8, 11, 13):
        vals = np.random.default_rng(7).integers(0, 1 << w, 5000).astype(np.uint16)
        words, _ = _pack_bits(vals, w)
        assert np.array_equal(_unpack_bits(words, len(vals), w), vals)
