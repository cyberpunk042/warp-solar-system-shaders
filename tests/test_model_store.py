"""Whole-model ChromoFold compression: stores for big tensors, lossless-vs-quant reconstruct, smaller than fp16."""
import numpy as np

import warp as wp

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _tiny_model():
    import torch
    import torch.nn as nn
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 400))


def test_compress_model_covers_big_tensors_only():
    from warp_compress.model_store import compress_model
    m = _tiny_model()
    stores, br = compress_model(m, bits=8, min_numel=50_000, device=_DEV)
    # the three Linear weights (>=50k) are stored; biases (small) are kept fp16
    assert len(stores) == 3
    assert br["big_params"] == 256 * 512 + 512 * 256 + 256 * 400


def test_reconstruct_into_is_lossless_vs_quant():
    import torch
    from warp_compress.model_store import compress_model, reconstruct_into
    from warp_compress.weight_store import QuantizedWeightStore
    m = _tiny_model()
    stores, _ = compress_model(m, bits=8, min_numel=50_000, device=_DEV)
    reconstruct_into(m, stores)
    for n, p in m.named_parameters():
        if n in stores:
            ref = QuantizedWeightStore(stores[n].reconstruct(), bits=8, huffman=True,
                                       device=_DEV)  # idempotent check below
            W = p.detach().numpy()
            # after reconstruct_into, the live weight equals its own store's reconstruction (bit-exact)
            assert np.array_equal(W, stores[n].reconstruct().reshape(W.shape))


def test_compressed_smaller_than_fp16():
    from warp_compress.model_store import compress_model
    m = _tiny_model()
    stores, br = compress_model(m, bits=4, min_numel=50_000, device=_DEV)
    fp16_big = br["big_params"] * 2
    assert br["compressed"] < fp16_big                   # int4 + entropy layer beats fp16 on the big tensors
    # (on tiny non-peaky tensors the fixed table/superblock overhead keeps b/w near int4; the fp16 win is the claim)
    assert br["compressed"] * 8 / br["big_params"] < 6.0


def test_protect_quantizes_named_tensors_at_higher_precision():
    from warp_compress.model_store import compress_model
    m = _tiny_model()
    # "0.weight" is the first Linear; protect it at int8 while the rest go int4
    stores, _ = compress_model(m, bits=4, min_numel=50_000, device=_DEV, protect=("0.",), protect_bits=8)
    assert stores["0.weight"].bits == 8                    # protected tensor kept at higher precision
    assert stores["2.weight"].bits == 4 and stores["4.weight"].bits == 4
    # mixed precision is bigger than all-int4 but smaller than all-int8
    all4 = compress_model(m, bits=4, min_numel=50_000, device=_DEV)[1]["compressed"]
    all8 = compress_model(m, bits=8, min_numel=50_000, device=_DEV)[1]["compressed"]
    mixed = compress_model(m, bits=4, min_numel=50_000, device=_DEV, protect=("0.",), protect_bits=8)[1]["compressed"]
    assert all4 < mixed < all8


def test_forward_changes_but_stays_finite_after_compression():
    import torch
    from warp_compress.model_store import compress_model, reconstruct_into
    m = _tiny_model()
    x = torch.randn(2, 256)
    with torch.no_grad():
        y0 = m(x).clone()
        stores, _ = compress_model(m, bits=8, min_numel=50_000, device=_DEV)
        reconstruct_into(m, stores)
        y1 = m(x)
    assert torch.isfinite(y1).all()
    assert not torch.allclose(y0, y1)                     # int8 quant perturbs the output (it's lossy)
    assert (y1 - y0).abs().max() < 0.5                    # but only slightly (int8 is close)
