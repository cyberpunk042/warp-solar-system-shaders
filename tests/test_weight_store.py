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


def test_group_quant_is_lossless_and_more_accurate():
    # a tensor with an outlier channel: per-tensor scale is forced coarse; group scales adapt
    rng = np.random.default_rng(8)
    W = (rng.standard_normal((256, 256)) * 0.02).astype(np.float32)
    W[0] *= 40.0                                          # one big-magnitude row (an outlier channel)
    per = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    grp = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, group_size=128)
    # both lossless vs their own quantization
    for st in (per, grp):
        lim = 7; pv = st._per_val_scale()
        refq = (np.clip(np.round(W.ravel() / pv), -lim, lim) * pv).reshape(W.shape)
        assert np.allclose(st.reconstruct(), refq, atol=1e-5)
    assert np.mean((grp.reconstruct() - W) ** 2) < np.mean((per.reconstruct() - W) ** 2)   # group is more accurate


def test_group_fetch_matches_reconstruct():
    W = _peaky((128, 128), seed=9)
    st = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, group_size=64)
    R = st.reconstruct().ravel()
    idx = np.random.default_rng(10).integers(0, st.n, 400)
    assert np.allclose(st.fetch(idx), R[idx])


def test_group_size_accounts_scale_side_channel():
    W = _peaky((256, 256), seed=11)
    st = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, group_size=128)
    assert st._scales is not None and st._scales.shape[0] == (W.size + 127) // 128
    assert st.size_bytes() > st.wm.index_bytes()          # includes the per-group scales
