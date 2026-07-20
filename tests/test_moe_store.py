"""MoE expert store: all experts entropy-coded in VRAM, only routed ones decoded, output lossless vs quant."""
import numpy as np

import warp as wp

from warp_compress.moe_store import MoEExpertStore, _silu
from warp_compress.weight_store import QuantizedWeightStore

_DEV = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"


def _experts(E=8, d=128, dff=128, seed=0):
    rng = np.random.default_rng(seed)
    seed_t = {k: (rng.standard_normal((a, b)) / np.sqrt(a)).astype(np.float32)
              for k, (a, b) in {"gate": (d, dff), "up": (d, dff), "down": (dff, d)}.items()}
    return [{k: v + rng.standard_normal(v.shape).astype(np.float32) * v.std() * 0.3 for k, v in seed_t.items()}
            for _ in range(E)], d, dff, E


def _ref_forward(experts, x, logits, k):
    d = experts[0]["down"].shape[1]
    topk = np.argsort(-logits, axis=1)[:, :k]
    w = np.take_along_axis(logits, topk, 1); w = np.exp(w - w.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
    rec = {int(e): {kk: QuantizedWeightStore(experts[int(e)][kk], bits=4, huffman=True, device=_DEV).reconstruct()
                    for kk in experts[0]} for e in np.unique(topk)}
    y = np.zeros((x.shape[0], d), np.float32)
    for slot in range(k):
        for e in np.unique(topk):
            m = topk[:, slot] == e
            if m.any():
                W = rec[int(e)]; h = _silu(x[m] @ W["gate"]) * (x[m] @ W["up"]); y[m] += w[m, slot][:, None] * (h @ W["down"])
    return y


def test_moe_output_is_lossless_vs_quantized():
    experts, d, dff, E = _experts()
    store = MoEExpertStore(experts, bits=4, huffman=True, device=_DEV)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((12, d)).astype(np.float32) * 0.5
    logits = rng.standard_normal((12, E)).astype(np.float32)
    y, _ = store.forward(x, logits, k=2)
    assert np.allclose(y, _ref_forward(experts, x, logits, 2), atol=1e-3)


def test_bank_is_smaller_than_dense_fp16():
    experts, *_ = _experts(E=8, d=256, dff=256)
    store = MoEExpertStore(experts, bits=4, huffman=True, device=_DEV)
    assert store.size_bytes() < store.dense_bytes()      # int4 + entropy layer << fp16
    assert store.size_bytes() * 8 / store._params < 4.0  # below fixed int4 too


def test_only_routed_experts_decoded():
    experts, d, dff, E = _experts(E=16)
    store = MoEExpertStore(experts, device=_DEV)
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, d)).astype(np.float32)
    logits = rng.standard_normal((4, E)).astype(np.float32)
    _, n_used = store.forward(x, logits, k=2)
    assert n_used <= 4 * 2                                # at most (tokens × k) distinct experts touched
    assert n_used < E                                    # and strictly fewer than the whole bank


def test_reconstruct_expert_is_bit_exact_vs_quant():
    experts, *_ = _experts(E=4)
    store = MoEExpertStore(experts, bits=4, huffman=True, device=_DEV)
    for e in range(4):
        R = store.reconstruct_expert(e)
        for k, W in experts[e].items():
            ref = QuantizedWeightStore(W, bits=4, huffman=True, device=_DEV).reconstruct()
            assert np.array_equal(R[k], ref)
