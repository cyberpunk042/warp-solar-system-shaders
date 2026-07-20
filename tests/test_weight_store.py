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


def _heavy_tailed(shape=(512, 256), frac=0.003, seed=0):
    # Gaussian bulk + a few large outliers (the SpQR regime: channels that blow the int4 scale)
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal(shape) / np.sqrt(shape[1])).astype(np.float32)
    oi = rng.choice(W.size, int(frac * W.size), replace=False)
    W.ravel()[oi] = (rng.standard_normal(oi.size) * 12 / np.sqrt(shape[1])).astype(np.float32)
    return W


def test_outliers_beat_plain_and_group_int4_on_accuracy():
    W = _heavy_tailed(seed=12)
    plain = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    grp = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, group_size=128)
    out = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, outliers=0.01)
    m = lambda st: np.mean((st.reconstruct() - W) ** 2)
    assert m(out) < m(grp) < m(plain)                     # the outlier side-channel fixes the cause -> best MSE
    assert out._out_idx is not None and out._out_idx.shape[0] == int(0.01 * W.size)


def test_outliers_are_exact_and_lossless_over_the_rest():
    W = _heavy_tailed(seed=13)
    st = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, outliers=0.01)
    R = st.reconstruct()
    assert np.array_equal(R.ravel()[st._out_idx], W.ravel()[st._out_idx].astype(np.float16).astype(np.float32))
    idx = np.random.default_rng(14).integers(0, st.n, 800)   # fetch (incl. outlier positions) == reconstruct
    assert np.allclose(st.fetch(idx), R.ravel()[idx])


def test_outliers_serialise_and_round_trip():
    W = _heavy_tailed(seed=15)
    st = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, outliers=0.01)
    st2 = QuantizedWeightStore.load(st.save(), device=_DEV)
    assert np.array_equal(st.reconstruct(), st2.reconstruct())
    assert st2._out_idx is not None and np.array_equal(st._out_idx, st2._out_idx)


def test_channel_scale_is_lossless_over_scaled_quant():
    from warp_compress.awq import awq_scale, _fake_quant
    W = _peaky((256, 384), seed=16)
    act = np.abs(np.random.default_rng(17).standard_normal(384)).astype(np.float32)
    s = awq_scale(W, act, bits=4)[0]
    st = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, channel_scale=s)
    cs = s.astype(np.float16).astype(np.float32)                # store rounds the scale to fp16
    ref = _fake_quant(W * cs[None, :], 4, None) / cs[None, :]   # dequant of scaled W, scale undone
    assert np.allclose(st.reconstruct(), ref, atol=1e-4)
    idx = np.random.default_rng(18).integers(0, W.size, 500)
    assert np.allclose(st.fetch(idx), st.reconstruct().ravel()[idx], atol=1e-5)   # random access still holds


def test_channel_scale_serialises_and_lowers_output_error():
    from warp_compress.awq import awq_scale
    rng = np.random.default_rng(19)
    W = (rng.standard_normal((256, 256)) / 16).astype(np.float32)
    act = np.abs(rng.standard_normal(256)).astype(np.float32)
    act[rng.choice(256, 5, replace=False)] *= 30.0             # salient channels -> AWQ should help
    s = awq_scale(W, act, bits=4)[0]
    awq = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV, channel_scale=s)
    plain = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV)
    x = act[None, :] * rng.standard_normal((32, 256)).astype(np.float32)
    err = lambda st: np.mean((x @ (W - st.reconstruct()).T) ** 2)
    assert err(awq) < err(plain)                               # activation-aware scaling lowers output error
    st2 = QuantizedWeightStore.load(awq.save(), device=_DEV)   # channel scale survives serialisation
    assert np.array_equal(awq.reconstruct(), st2.reconstruct())
