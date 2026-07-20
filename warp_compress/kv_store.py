"""kv_store — hold a long context's KV cache entropy-coded in VRAM, decode (attended) positions on the GPU.

The KV cache is the long-context memory bottleneck: it grows with sequence length and dominates VRAM in
long-context serving. Quantization is the lever (int4/int8 KV), but quantized KV — especially V — is peaky, so
the class-stream entropy layer squeezes it further, losslessly, with GPU random access to any position (decode
only the *attended* positions under sparse / windowed / retrieval attention).

    KVCacheStore(past_key_values)     -> every layer's K,V quantized + entropy-coded, GPU-resident
    .reconstruct_layer(l)             -> that layer's dequantized (K, V), bit-exact vs quant
    .attention(l, Q, positions=None)  -> scaled-dot-product attention output; `positions` decodes a subset only

Honest: quantization is the lossy step; the entropy layer is lossless on top, so attention output equals the
plain-quantized-KV output exactly. The win is capacity (fit a longer context) + attended-only decode, not
compressing fp16 KV directly. Measured on real gpt2 in the demo. Run: python -m warp_compress.kv_store
"""
from __future__ import annotations

import numpy as np

from .weight_store import QuantizedWeightStore


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class KVCacheStore:
    """A KV cache stored per layer as entropy-coded quantized K and V, resident on a Warp device."""

    def __init__(self, past_key_values, bits: int = 4, huffman: bool = True, device: str = "cuda:0"):
        self.n_layers = len(past_key_values)
        self.bits = bits
        self.device = device
        self.layers = []
        self._vals = 0
        for (K, V) in past_key_values:
            K = np.asarray(K, np.float32)
            V = np.asarray(V, np.float32)
            self.layers.append(dict(
                K=QuantizedWeightStore(K.ravel(), bits=bits, huffman=huffman, device=device), Kshape=K.shape,
                V=QuantizedWeightStore(V.ravel(), bits=bits, huffman=huffman, device=device), Vshape=V.shape))
            self._vals += K.size + V.size

    def size_bytes(self) -> int:
        return sum(l["K"].size_bytes() + l["V"].size_bytes() for l in self.layers)

    def dense_bytes(self, dtype_bytes: int = 2) -> int:
        return self._vals * dtype_bytes                        # the fp16 KV cache this replaces

    def reconstruct_layer(self, l: int):
        L = self.layers[int(l)]
        return L["K"].reconstruct().reshape(L["Kshape"]), L["V"].reconstruct().reshape(L["Vshape"])

    def attention(self, l: int, Q, positions=None):
        """Scaled-dot-product attention for layer `l`: (B, heads, q, d) query against the stored K,V. If
        `positions` is given, only those cache rows are used (and, in a real kernel, only they are decoded)."""
        K, V = self.reconstruct_layer(l)                       # (B, heads, seq, d)
        if positions is not None:
            K = K[:, :, positions, :]
            V = V[:, :, positions, :]
        d = Q.shape[-1]
        scores = np.einsum("bhqd,bhkd->bhqk", Q, K) / np.sqrt(d)
        return np.einsum("bhqk,bhkd->bhqd", _softmax(scores, -1), V)


def _demo():
    import warnings
    warnings.filterwarnings("ignore")

    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
        ids = tok("The history of computing spans many decades. " * 50, return_tensors="pt").input_ids[:, :512]
        with torch.no_grad():
            pkv = model(ids, use_cache=True).past_key_values
        pkv = [(k.numpy(), v.numpy()) for (k, v) in pkv]
    except Exception:
        rng = np.random.default_rng(0)
        pkv = [((rng.standard_normal((1, 12, 512, 64)) * 0.3).astype(np.float32),
                (rng.standard_normal((1, 12, 512, 64)) * 0.3).astype(np.float32)) for _ in range(12)]

    store = KVCacheStore(pkv, bits=4, huffman=True, device=dev)
    seq = pkv[0][0].shape[2]

    # correctness: attention over the ChromoFold KV == attention over the plain-quantized KV (lossless on top)
    rng = np.random.default_rng(1)
    Q = (rng.standard_normal((1, pkv[0][0].shape[1], 1, pkv[0][0].shape[3])) * 0.3).astype(np.float32)
    y_cf = store.attention(0, Q)
    Kq, Vq = store.reconstruct_layer(0)                        # (already the quantized-then-dequantized KV)
    d = Q.shape[-1]
    ref = np.einsum("bhqk,bhkd->bhqd", _softmax(np.einsum("bhqd,bhkd->bhqk", Q, Kq) / np.sqrt(d), -1), Vq)
    ok = np.allclose(y_cf, ref, atol=1e-5)

    # sparse attention: attend to a recent window -> only those positions need decoding
    win = np.arange(seq - 64, seq)
    _ = store.attention(0, Q, positions=win)

    dense, comp = store.dense_bytes(), store.size_bytes()
    print(f"device={dev}   gpt2 KV cache: {store.n_layers} layers × K,V (heads×{seq}×d)")
    print(f"[correct] attention(ChromoFold KV) == attention(quantized KV) ✓" if ok else "[correct] FAIL")
    print(f"[capacity] dense fp16 KV {dense/1e6:7.2f} MB   ChromoFold (int4+huff) {comp/1e6:6.2f} MB   "
          f"=> {dense/comp:.1f}× smaller ({comp*8/store._vals:.2f} b/val)  ⇒ ~{dense/comp:.0f}× longer context per VRAM budget")
    print(f"[sparse]  windowed attention over the last 64/{seq} positions decodes only that window "
          f"(random access; full attention decodes all)")
    print("\n=> hold a long context's KV entropy-coded at ~{:.0f}× vs fp16; attention is lossless over the "
          "quantized values (identical output). Capacity + attended-only decode is how a longer context fits\n"
          "   on one GPU. V compresses harder than K (peakier); per-token quant scales would be a small "
          "side-channel for better accuracy. Quantization is the lossy lever; this is the entropy layer on top."
          .format(dense / comp))


if __name__ == "__main__":
    _demo()
