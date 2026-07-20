"""KV cache store: entropy-coded quantized K/V in VRAM, attention lossless vs quant, attended-subset decode."""
import numpy as np

import warp as wp

from warp_compress.kv_store import KVCacheStore, _softmax

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _kv(layers=4, heads=8, seq=128, d=32, seed=0):
    rng = np.random.default_rng(seed)
    return [((rng.standard_normal((1, heads, seq, d)) * 0.3).astype(np.float32),
             (rng.standard_normal((1, heads, seq, d)) * 0.3).astype(np.float32)) for _ in range(layers)]


def test_attention_is_lossless_vs_quantized_kv():
    pkv = _kv(seed=1)
    store = KVCacheStore(pkv, bits=4, huffman=True, device=_DEV)
    rng = np.random.default_rng(2)
    Q = (rng.standard_normal((1, 8, 1, 32)) * 0.3).astype(np.float32)
    y = store.attention(0, Q)
    K, V = store.reconstruct_layer(0)
    d = Q.shape[-1]
    ref = np.einsum("bhqk,bhkd->bhqd", _softmax(np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(d), -1), V)
    assert np.allclose(y, ref, atol=1e-5)


def test_kv_smaller_than_dense_fp16():
    store = KVCacheStore(_kv(seed=3), bits=4, device=_DEV)
    assert store.size_bytes() < store.dense_bytes()      # well below fp16 (int4 + entropy, + per-axis scales)
    assert store.size_bytes() * 8 / store._vals < 6.0


def test_reconstruct_layer_roundtrips_its_quantization():
    pkv = _kv(layers=3, seed=4)
    store = KVCacheStore(pkv, device=_DEV)
    for l in range(3):
        K, V = store.reconstruct_layer(l)
        assert K.shape == pkv[l][0].shape and V.shape == pkv[l][1].shape
        K2, V2 = store.reconstruct_layer(l)              # decode is deterministic + lossless over the quant
        assert np.array_equal(K, K2) and np.array_equal(V, V2)
        assert np.mean((K - pkv[l][0]) ** 2) < 1e-2      # int4 dequant is close to the original


def test_per_axis_is_more_accurate_than_per_tensor():
    # a KV tensor with a Key outlier channel: per-channel scaling should reduce error vs per-tensor
    rng = np.random.default_rng(7)
    K = (rng.standard_normal((1, 8, 64, 32)) * 0.3).astype(np.float32)
    K[:, :, :, 3] *= 12.0                                 # one outlier channel
    V = (rng.standard_normal((1, 8, 64, 32)) * 0.3).astype(np.float32)
    pkv = [(K, V)]
    per_t = KVCacheStore(pkv, bits=4, per_axis=False, device=_DEV)
    per_a = KVCacheStore(pkv, bits=4, per_axis=True, device=_DEV)
    Kt, _ = per_t.reconstruct_layer(0)
    Ka, _ = per_a.reconstruct_layer(0)
    assert np.mean((Ka - K) ** 2) < np.mean((Kt - K) ** 2)   # per-channel tames the outlier -> lower error


def test_windowed_attention_matches_full_on_that_window():
    pkv = _kv(seq=96, seed=5)
    store = KVCacheStore(pkv, device=_DEV)
    Q = (np.random.default_rng(6).standard_normal((1, 8, 1, 32)) * 0.3).astype(np.float32)
    win = np.arange(96 - 32, 96)
    y_win = store.attention(0, Q, positions=win)
    K, V = store.reconstruct_layer(0)                    # full attention restricted to the same window
    d = Q.shape[-1]
    sc = np.einsum("bhqd,bhkd->bhqk", Q, K[:, :, win, :]) / np.sqrt(d)
    ref = np.einsum("bhqk,bhkd->bhqd", _softmax(sc, -1), V[:, :, win, :])
    assert np.allclose(y_win, ref, atol=1e-5)
