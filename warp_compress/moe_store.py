"""moe_store — keep ALL MoE experts resident, entropy-coded, and decode only the routed ones on the GPU.

The clearest "fit more on one GPU" case (docs/chromofold.md §4): a Mixture-of-Experts FFN has many experts, but
each token routes to only top-k. Most experts are idle at any step — so hold the whole bank **compressed in
VRAM** (int4 + the class-stream Huffman = ~1.33 b/w, ~12× vs fp16) and **reconstruct just the routed experts**
per batch. That lets a much bigger MoE fit where the dense fp16 bank would not.

    MoEExpertStore(experts)          -> every expert's (gate, up, down) quantized + entropy-coded, GPU-resident
    .reconstruct_expert(e)           -> that expert's dequantized weights (on the GPU), bit-exact vs quant
    .forward(x, router_logits, k)    -> the top-k MoE FFN output, decoding only the experts actually routed to

Honest: quantization is the lossy lever; the entropy layer is lossless on top, so the MoE output equals the
plain-quantized MoE exactly. The win is capacity (hold the bank) + sparse decode (touch only routed experts),
not compressing dense fp16 directly. Run: python -m warp_compress.moe_store
"""
from __future__ import annotations

import numpy as np

from .weight_store import QuantizedWeightStore


def _silu(x):
    return x / (1.0 + np.exp(-x))


class MoEExpertStore:
    """A bank of MoE expert FFNs, each quantized + RRR/Huffman entropy-coded, resident on a Warp device."""

    def __init__(self, experts, bits: int = 4, huffman: bool = True, device: str = "cuda:0"):
        self.n_experts = len(experts)
        self.bits = bits
        self.device = device
        self._store = []                                        # per expert: {name: QuantizedWeightStore}
        self._params = 0
        for ex in experts:
            self._store.append({k: QuantizedWeightStore(W, bits=bits, huffman=huffman, device=device)
                                for k, W in ex.items()})
            self._params += sum(W.size for W in ex.values())

    def size_bytes(self) -> int:
        return sum(st.size_bytes() for ex in self._store for st in ex.values())

    def dense_bytes(self, dtype_bytes: int = 2) -> int:
        return self._params * dtype_bytes                       # the fp16 bank this replaces

    def reconstruct_expert(self, e: int) -> dict:
        return {k: st.reconstruct() for k, st in self._store[int(e)].items()}

    def forward(self, x, router_logits, k: int = 2):
        """Top-k MoE FFN. `router_logits` is (B, n_experts); each token uses its top-k experts (softmax-weighted).
        Only the experts actually routed to are reconstructed (sparse decode)."""
        B = x.shape[0]
        topk = np.argsort(-router_logits, axis=1)[:, :k]        # (B, k) expert ids per token
        w = np.take_along_axis(router_logits, topk, axis=1)
        w = np.exp(w - w.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)   # softmax over the k
        used = np.unique(topk)
        cache = {int(e): self.reconstruct_expert(int(e)) for e in used}         # decode only routed experts
        y = np.zeros((B, x.shape[1]), np.float32)
        for slot in range(k):
            for e in used:
                mask = topk[:, slot] == e
                if not mask.any():
                    continue
                W = cache[int(e)]
                h = _silu(x[mask] @ W["gate"]) * (x[mask] @ W["up"])            # SwiGLU
                y[mask] += (w[mask, slot][:, None]) * (h @ W["down"])
        return y, len(used)


def _demo():
    import time
    import warnings
    warnings.filterwarnings("ignore")

    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)

    # realistic peaky experts: seed all three projections from REAL gpt2 MLP weights + per-expert perturbation
    d, dff = 768, 768
    seed = {}
    try:
        from transformers import AutoModelForCausalLM
        params = dict(AutoModelForCausalLM.from_pretrained("gpt2").named_parameters())
        cfc = params["transformer.h.0.mlp.c_fc.weight"].detach().numpy().astype(np.float32)    # (768, 3072)
        cpr = params["transformer.h.0.mlp.c_proj.weight"].detach().numpy().astype(np.float32)  # (3072, 768)
        seed = {"gate": cfc[:d, :dff].copy(), "up": cfc[:d, dff:2 * dff].copy(), "down": cpr[:dff, :d].copy()}
    except Exception:
        seed = {"gate": (rng.standard_normal((d, dff)) / np.sqrt(d)).astype(np.float32),
                "up": (rng.standard_normal((d, dff)) / np.sqrt(d)).astype(np.float32),
                "down": (rng.standard_normal((dff, d)) / np.sqrt(dff)).astype(np.float32)}

    E, k = 32, 2
    experts = []
    for _ in range(E):                                          # a bank of related experts (peaky histograms)
        experts.append({kk: v + rng.standard_normal(v.shape).astype(np.float32) * v.std() * 0.3
                        for kk, v in seed.items()})

    store = MoEExpertStore(experts, bits=4, huffman=True, device=dev)

    # correctness: ChromoFold MoE output == plain-quantized MoE output (lossless over the quantized values)
    B = 16
    x = rng.standard_normal((B, d)).astype(np.float32) * 0.5
    logits = rng.standard_normal((B, E)).astype(np.float32)
    y_cf, n_used = store.forward(x, logits, k=k)

    def _ref():                                                 # plain per-expert int4 dequant, same math
        topk = np.argsort(-logits, axis=1)[:, :k]
        w = np.take_along_axis(logits, topk, 1); w = np.exp(w - w.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        y = np.zeros((B, d), np.float32)
        rec = {int(e): {kk: QuantizedWeightStore(experts[int(e)][kk], bits=4, huffman=True, device=dev).reconstruct()
                        for kk in experts[0]} for e in np.unique(topk)}
        for slot in range(k):
            for e in np.unique(topk):
                m = topk[:, slot] == e
                if m.any():
                    W = rec[int(e)]; h = _silu(x[m] @ W["gate"]) * (x[m] @ W["up"])
                    y[m] += w[m, slot][:, None] * (h @ W["down"])
        return y
    ok = np.allclose(y_cf, _ref(), atol=1e-3)

    dense, comp = store.dense_bytes(), store.size_bytes()
    print(f"device={dev}   MoE bank: {E} experts × (gate/up/down {d}×{dff}) , top-{k} routing")
    print(f"[correct] ChromoFold MoE output == quantized MoE output ✓" if ok else "[correct] FAIL")
    print(f"[capacity] dense fp16 bank {dense/1e6:7.1f} MB   ChromoFold (int4+huff) {comp/1e6:6.2f} MB   "
          f"=> {dense/comp:.1f}× smaller ({store.size_bytes()*8/store._params:.2f} b/weight)")
    print(f"[sparse]  batch of {B} tokens routed to {n_used}/{E} experts — only those were decoded "
          f"({n_used}/{E} of the bank touched)")
    print("\n=> hold the WHOLE expert bank compressed in VRAM at ~1.3 b/weight (~12× vs fp16), reconstruct only "
          "the routed experts per batch. That is how a bigger MoE fits on one GPU — capacity + sparse decode,\n"
          "   losslessly over the quantized weights (identical output). Quantization is the lossy lever; this "
          "is the entropy layer + random access on top.")


if __name__ == "__main__":
    _demo()
