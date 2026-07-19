"""lora_library — ChromoFold as a hot-swappable LoRA/adapter library, decoded on the GPU.

The clearest "fit more on one GPU" story in `docs/chromofold.md` §4/§5: a *library* of task adapters, held as
one base + a tree of sparse deltas, reconstructed in VRAM on demand. A LoRA adapter is a low-rank pair (A, B)
with ΔW = B·A; a family of sibling/checkpoint adapters is near-duplicate, so on a shared int quant grid they
differ in only a few entries — exactly the reference-delta regime (`gpu_delta.GPUDeltaCluster`).

This harness is honest about its setting: there is no torch/transformers in this environment, so it is NOT a
live HF checkpoint. It IS faithful where it counts — real low-rank factors, real int8 quantization on a shared
grid, the exact `gpu_delta` store, real forward passes — and the reconstruction is *lossless vs the quantized
adapter*, so a torch/HF wiring is the same store→reconstruct with tensors instead of numpy.

    ChromoLoRALibrary(base_factors, family)  -> stores the quantized family as a GPU delta cluster
    .reconstruct(k)                          -> adapter k's (A, B), bit-exact vs its quantized form, on GPU
    .apply(x, k, W, scale)                   -> a forward pass y = x·(W + scale·B·A)ᵀ with the swapped adapter

Run: python -m warp_compress.lora_library
"""
from __future__ import annotations

import numpy as np

from .gpu_delta import GPUDeltaCluster


def quantize_shared(tensors, bits: int = 8):
    """Per-family int quantization on ONE shared grid (a fixed quant scale), so near-duplicate float adapters
    stay near-duplicate as ints — the property the delta tree needs. Returns (int8 stack, scale)."""
    lim = (1 << (bits - 1)) - 1                                    # 127 for int8
    scale = float(np.max([np.abs(t).max() for t in tensors])) / lim + 1e-12
    q = np.stack([np.clip(np.round(t / scale), -lim, lim).astype(np.int64) for t in tensors])
    return q, scale


def synth_family(d_out=1024, d_in=1024, r=16, k=64, perturb=0.02, seed=0):
    """A base weight W and a family of `k` related LoRA adapters (a shared ancestor + sparse per-task edits) —
    the checkpoint-series / sibling-fine-tune regime where a delta library pays off."""
    rng = np.random.default_rng(seed)
    W = (rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)
    A0 = (rng.standard_normal((r, d_in)) / np.sqrt(d_in)).astype(np.float32)   # shared ancestor factors
    B0 = (rng.standard_normal((d_out, r)) / np.sqrt(r)).astype(np.float32)
    As, Bs = [], []
    for _ in range(k):
        A = A0.copy(); B = B0.copy()
        fa = rng.integers(0, A.size, int(perturb * A.size))       # a few per-task edits (sparse divergence)
        fb = rng.integers(0, B.size, int(perturb * B.size))
        A.flat[fa] += rng.standard_normal(fa.size).astype(np.float32) * A.std() * 0.5
        B.flat[fb] += rng.standard_normal(fb.size).astype(np.float32) * B.std() * 0.5
        As.append(A); Bs.append(B)
    return W, As, Bs


class ChromoLoRALibrary:
    """A quantized LoRA family stored as ONE base + sparse deltas, reconstructed adapter-by-adapter on the GPU."""

    def __init__(self, As, Bs, device: str = "cuda:0"):
        self.k = len(As)
        self.rshape = As[0].shape
        self.bshape = Bs[0].shape
        self._na = As[0].size
        self.Aq, self.sa = quantize_shared(As)                    # int8 on shared grids
        self.Bq, self.sb = quantize_shared(Bs)
        # each adapter = one flat int sequence [Bq | Aq]; the family is a near-duplicate cluster
        seqs = [np.concatenate([self.Bq[i].ravel(), self.Aq[i].ravel()]) for i in range(self.k)]
        self.cluster = GPUDeltaCluster(seqs, device=device)
        self._raw_int8 = sum(s.shape[0] for s in seqs)            # bytes if every adapter stored independently

    def size_bytes(self) -> int:
        return self.cluster.size_bytes()

    def reconstruct(self, k: int):
        """Adapter k's dequantized (A, B), decoded on the GPU — bit-exact vs its quantized form."""
        flat = self.cluster.decode_leaf(int(k))
        B = flat[: self.bshape[0] * self.bshape[1]].reshape(self.bshape).astype(np.float32) * self.sb
        A = flat[self.bshape[0] * self.bshape[1]:].reshape(self.rshape).astype(np.float32) * self.sa
        return A, B

    def apply(self, x, k: int, W, scale: float = 1.0):
        A, B = self.reconstruct(k)
        return x @ (W + scale * (B @ A)).T


def _demo():
    import time

    dev_ok = True
    try:
        import warp as wp
        dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        dev, dev_ok = "cpu", False

    W, As, Bs = synth_family(k=64, perturb=0.02)
    lib = ChromoLoRALibrary(As, Bs, device=dev)

    # 1) correctness: reconstruction is lossless vs the quantized adapter, so the forward pass matches exactly
    x = np.random.default_rng(1).standard_normal((8, W.shape[1])).astype(np.float32)
    ok = True
    for k in (0, 17, 40, 63):
        A, B = lib.reconstruct(k)
        refA = lib.Aq[k].astype(np.float32) * lib.sa                # the quantized adapter it should equal
        refB = lib.Bq[k].astype(np.float32) * lib.sb
        ok &= np.array_equal(A, refA) and np.array_equal(B, refB)
        y = lib.apply(x, k, W)
        yref = x @ (W + (refB @ refA)).T
        ok &= np.allclose(y, yref, atol=1e-4)

    # 2) the library economics — hold the whole family resident, on one GPU
    fp32 = sum(a.nbytes + b.nbytes for a, b in zip(As, Bs))
    int8 = lib._raw_int8
    chromo = lib.size_bytes()
    t0 = time.perf_counter(); [lib.reconstruct(k) for k in range(lib.k)]; dt = time.perf_counter() - t0

    print(f"device={dev}   LoRA family: {lib.k} adapters, factors A{lib.rshape} B{lib.bshape}, ~2% divergence")
    print(f"[correct] GPU reconstruct == quantized adapter (bit-exact) ✓   forward pass matches ✓" if ok
          else "[correct] FAIL")
    print(f"[size]  fp32 independent {fp32/1e6:7.2f} MB   int8 independent {int8/1e6:6.2f} MB   "
          f"ChromoFold {chromo/1e6:6.3f} MB")
    print(f"        => {int8/chromo:5.1f}× vs int8,  {fp32/chromo:5.1f}× vs fp32  "
          f"(so ~{int8/chromo:.0f}× more adapters fit in the same VRAM budget)")
    print(f"[speed] reconstructed all {lib.k} adapters in {dt*1e3:.1f} ms "
          f"({lib.k/dt:,.0f} adapters/s, on the {dev})")
    print("=> a whole LoRA/adapter library lives compressed in VRAM as base + sparse deltas; any adapter "
          "hot-swaps by a GPU delta-decode. This is docs/chromofold.md §5's adapter-library win, measured.\n"
          "   (Controlled harness — no torch/HF in this env; the store→reconstruct is identical with tensors.)")


if __name__ == "__main__":
    _demo()
