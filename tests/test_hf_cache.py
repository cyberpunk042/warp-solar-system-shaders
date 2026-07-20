"""ChromoFold as a drop-in transformers Cache: compresses the settled prefix, keeps an fp16 residual window."""
import numpy as np
import pytest

import warp as wp

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from warp_compress.hf_cache import make_cache

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def test_prefix_compresses_and_length_tracks():
    cache = make_cache(residual=16, bits=4, device=_DEV)
    H, d = 2, 64
    for _ in range(80):                                  # 80 single-token decode steps
        K, V = cache.update(torch.randn(1, H, 1, d), torch.randn(1, H, 1, d), 0)
    assert int(cache.layers[0].get_seq_length()) == 80
    assert K.shape[-2] == 80 and V.shape[-2] == 80       # full sequence reassembled for attention
    assert cache.layers[0]._settled >= 80 - 2 * 16       # everything past the residual window is compressed
    fp16 = 80 * H * d * 2 * 2                             # K+V, fp16
    assert cache.memory_bytes() < fp16                   # resident KV is below the equivalent fp16 cache


def test_prefill_then_decode_stays_finite_and_full_length():
    cache = make_cache(residual=8, bits=4, device=_DEV)
    torch.manual_seed(0)
    K, V = cache.update(torch.randn(1, 2, 200, 64) * 0.3, torch.randn(1, 2, 200, 64) * 0.3, 0)   # prefill
    assert K.shape[-2] == 200
    for _ in range(10):                                  # then decode
        K, V = cache.update(torch.randn(1, 2, 1, 64) * 0.3, torch.randn(1, 2, 1, 64) * 0.3, 0)
    assert K.shape[-2] == 210 and torch.isfinite(K).all() and torch.isfinite(V).all()
    assert cache.layers[0]._settled > 0                  # the prefill prefix got compressed


def test_reassembled_prefix_is_the_quantized_roundtrip():
    # the compressed prefix must reconstruct to its KIVI-quantized values, not garbage
    from warp_compress.kv_store import KVCacheStore
    cache = make_cache(residual=4, bits=4, device=_DEV)
    torch.manual_seed(1)
    Kin = torch.randn(1, 2, 60, 64) * 0.3
    Vin = torch.randn(1, 2, 60, 64) * 0.3
    K, _ = cache.update(Kin, Vin, 0)
    # the first compressed chunk should equal a standalone KVCacheStore of the same flushed tokens
    st = cache.layers[0]._chunks[0]
    Kr, _ = st.reconstruct_layer(0)
    ref = KVCacheStore([(Kin[:, :, :Kr.shape[2], :].numpy(), Vin[:, :, :Kr.shape[2], :].numpy())],
                       bits=4, device=_DEV, per_axis=True).reconstruct_layer(0)[0]
    assert np.allclose(Kr, ref, atol=1e-4)
