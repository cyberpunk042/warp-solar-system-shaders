"""sparse_store — 2:4 semi-structured sparsity (the hardware lever), measured honestly.

The one lever category ChromoFold had not touched is structured sparsity. NVIDIA Ampere+ tensor cores run a
**2:4** pattern — exactly 2 of every 4 contiguous weights are nonzero — at ~2x throughput. This implements it
as a codec: magnitude-prune each group of 4 to its two largest, store the surviving values (scalar int,
entropy-coded) + a 3-bit **pattern id** per group (which 2 of the 4 survived; C(4,2)=6 patterns). Both streams
go in the `RRRWaveletGPU` self-index, so it stays GPU-addressable; the lossy lever is the pruning + the value
quantization.

**Read this honestly.** Post-training 2:4 magnitude pruning **drops half the weights**, so without fine-tuning
it costs real accuracy — at matched bits it is *worse* than dense int/PQ, and the demo shows exactly that. Its
value is **not** ratio or quality: it is the 2x sparse-tensor-core **throughput** on supported hardware (not
measurable in this CPU container) plus a structural-sparsity option that stays addressable. Shipping it as a
measured **negative** is the point — it maps the lever, and the number tells you the accuracy you'd have to buy
back with fine-tuning to use the hardware speedup.

    Sparse24WeightStore(W, value_bits=4)   -> pruned+quantized values + pattern ids, GPU-addressable
    .reconstruct()  ;  .fetch(flat_indices)  ;  .bits_per_weight()  ;  .sparsity  (== 0.5, exactly 2:4)

Measured: `python -m warp_compress.sparse_store`. Real-model perplexity (and the sparse-core speedup) are the
follow-up on the appropriate hardware — see docs/research/47.
"""
from __future__ import annotations

import numpy as np

from .gpu_rrr_wavelet import RRRWaveletGPU

# the 6 ways to keep 2 of 4 positions -> a pattern id in [0,6); pos_of[id] = the two kept lane indices
_PATTERNS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_PAT_ARR = np.array(_PATTERNS, np.int64)                        # (6, 2)


def _pair_to_id(a, b):
    lut = {p: i for i, p in enumerate(_PATTERNS)}
    return np.array([lut[(int(min(x, y)), int(max(x, y)))] for x, y in zip(a, b)], np.int64)


class Sparse24WeightStore:
    """2:4 magnitude-pruned + int-quantized weights; surviving values + pattern ids in the RRR self-index."""

    def __init__(self, W, value_bits: int = 4, device: str = "cuda:0"):
        W = np.asarray(W, np.float32)
        self.shape = W.shape
        self.n = int(W.size)
        self.value_bits = int(value_bits)
        self.device = device
        if self.n % 4 != 0:
            raise ValueError(f"tensor size {self.n} must be a multiple of 4 for 2:4 sparsity")
        g = W.ravel().reshape(-1, 4)                            # (n/4, 4) groups
        self.n_groups = g.shape[0]
        keep = np.argsort(-np.abs(g), axis=1)[:, :2]            # indices of the two largest-|.| per group
        keep.sort(axis=1)                                       # canonical order -> stable pattern id
        self.pattern = _pair_to_id(keep[:, 0], keep[:, 1])      # (n/4,) in [0,6)
        vals = np.take_along_axis(g, keep, axis=1).ravel()      # (n/2,) surviving values, in lane order
        lim = (1 << (self.value_bits - 1)) - 1
        self.scale = float(np.abs(vals).max()) / lim + 1e-12
        self._zero = lim
        q = (np.clip(np.round(vals / self.scale), -lim, lim).astype(np.int64) + lim)
        self.val_wm = RRRWaveletGPU(q, device=device, bits=self.value_bits)      # entropy-coded, addressable
        self.pat_wm = RRRWaveletGPU(self.pattern, device=device, bits=3)         # 6 patterns -> 3 bits, skewed
        self.sparsity = 0.5                                     # exactly 2:4

    def size_bytes(self) -> int:
        return int(self.val_wm.index_bytes() + self.pat_wm.index_bytes() + 8)    # + fp16 scale (8 rounded)

    def bits_per_weight(self) -> float:
        return self.size_bytes() * 8 / self.n

    def _values(self):
        q = self.val_wm.access(np.arange(self.n_groups * 2, dtype=np.int64))
        return (q - self._zero).astype(np.float32) * self.scale

    def reconstruct(self) -> np.ndarray:
        pat = self.pat_wm.access(np.arange(self.n_groups, dtype=np.int64))
        vals = self._values().reshape(self.n_groups, 2)
        out = np.zeros((self.n_groups, 4), np.float32)
        lanes = _PAT_ARR[pat]                                   # (n_groups, 2) kept lane indices
        np.put_along_axis(out, lanes, vals, axis=1)            # scatter survivors; the other two stay 0
        return out.reshape(self.shape).astype(np.float32)

    def fetch(self, flat_indices) -> np.ndarray:
        idx = np.asarray(flat_indices, np.int64)
        grp = idx // 4
        lane = idx % 4
        pat = self.pat_wm.access(grp)
        lanes = _PAT_ARR[pat]                                   # (m, 2) kept lanes for each queried group
        # value index within the flat value stream: 2 per group; slot 0 or 1 if this lane survived, else zero
        slot = np.where(lanes[:, 0] == lane, 0, np.where(lanes[:, 1] == lane, 1, -1))
        out = np.zeros(idx.shape[0], np.float32)
        hit = slot >= 0
        vflat = (grp * 2 + slot)[hit]
        q = self.val_wm.access(vflat.astype(np.int64))
        out[hit] = (q - self._zero).astype(np.float32) * self.scale
        return out


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    from .weight_store import QuantizedWeightStore
    from .vq_store import ProductQuantizedWeightStore
    rng = np.random.default_rng(0)
    d_out, d_in = 2048, 512
    W = (rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)
    x = rng.standard_normal((64, d_in)).astype(np.float32)
    fp32_out = x @ W.T

    def report(label, st):
        R = st.reconstruct()
        mse = float(np.mean((R - W) ** 2))
        oerr = float(np.mean((x @ R.T - fp32_out) ** 2))
        print(f"  {label:>24} {st.bits_per_weight():>9.2f} {mse:>12.2e} {oerr:>12.2e}")

    print(f"device={dev}   Gaussian weight tensor {W.shape} (no fine-tuning — post-training only)\n")
    print(f"  {'config':>24} {'b/weight':>9} {'MSE vs fp32':>12} {'out-err':>12}")
    report("dense int8", QuantizedWeightStore(W, bits=8, huffman=True, device=dev))
    report("dense int4", QuantizedWeightStore(W, bits=4, huffman=True, device=dev))
    report("dense PQ 4x8", ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=dev))
    for vb in (8, 4):
        st = Sparse24WeightStore(W, value_bits=vb, device=dev)
        idx = rng.integers(0, W.size, 3000)                    # fetch == reconstruct -> addressable + lossless/quant
        ok = np.allclose(st.fetch(idx), st.reconstruct().ravel()[idx], atol=1e-6)
        report(f"2:4 int{vb} {'✓' if ok else 'FAIL'}", st)
    print("\n=> HONEST NEGATIVE: post-training 2:4 magnitude pruning drops half the weights, so at matched bits it\n"
          "   is WORSE than dense int/PQ (2:4-int8 spends ~4.8 b/w yet its output error is far above dense int4's\n"
          "   at 2.9 b/w) — the dropped 50% is simply gone. 2:4 is NOT a ratio or quality win here; its payoff is\n"
          "   the ~2x sparse-tensor-core THROUGHPUT on supported GPUs (not measurable on CPU), which you would\n"
          "   pair with fine-tuning to recover the accuracy this number quantifies. The codec stays addressable\n"
          "   (values + 3-bit pattern ids in the RRR index); the lever is mapped, with its real cost measured.")


if __name__ == "__main__":
    _demo()
