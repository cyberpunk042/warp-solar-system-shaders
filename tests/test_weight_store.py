"""Quantized weight store: entropy-code quantized weights, lossless over the quantized values, GPU-addressable."""
import numpy as np

import warp as wp

from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _peaky(shape=(512, 256), seed=0):
    # Gaussian weights -> peaky histogram after quantization (the real-weight regime)
    return (np.random.default_rng(seed).standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)


def _quant_dequant(W, bits, scale):
    lim = (1 << (bits - 1)) - 1
    return (np.clip(np.round(W / scale), -lim, lim) * scale).astype(np.float32)


def test_reconstruct_is_lossless_over_quantized_values():
    W = _peaky(seed=1)
    for bits in (4, 8):
        st = QuantizedWeightStore(W, bits=bits, device=_DEV)
        assert np.array_equal(st.reconstruct(), _quant_dequant(W, bits, st.scale))   # bit-exact vs plain-quant


def test_fetch_matches_reconstruct():
    W = _peaky(seed=2)
    st = QuantizedWeightStore(W, bits=4, device=_DEV)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(3).integers(0, st.n, 500)
    assert np.allclose(st.fetch(idx), R[idx])


def test_int4_beats_fixed_width_on_peaky_weights():
    W = _peaky((1024, 512), seed=4)
    st = QuantizedWeightStore(W, bits=4, device=_DEV)
    assert st.bits_per_weight() < 4.0                    # entropy layer compresses below fixed int4


def test_shape_and_scale_preserved():
    W = _peaky((300, 200), seed=5)
    st = QuantizedWeightStore(W, bits=8, device=_DEV)
    R = st.reconstruct()
    assert R.shape == W.shape
    assert np.mean((R - W) ** 2) < 1e-3                  # int8 dequant is close to the original
