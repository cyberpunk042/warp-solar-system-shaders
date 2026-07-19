"""weight_store — entropy-code QUANTIZED model weights with GPU random access. Combine with quantization.

The vision doc's core "fit more on one GPU" claim for weights (§4): quantization is the big lever (int4/int8),
but quantized weights are NOT uniform — their histogram is peaky (weights cluster near zero) — so an entropy
coder squeezes them further, losslessly, on top of quantization. And because it's the RRR wavelet, any weight
is randomly addressable on the GPU: fetch the routed MoE expert / dequant on the fly, without materialising the
rest.

    QuantizedWeightStore(W, bits=4)   -> quantize + RRR-entropy-code, resident on the GPU
    .reconstruct()                    -> the dequantized tensor (bit-exact vs its quantized form)
    .fetch(flat_indices)              -> specific weights, decoded on the GPU
    .size_bytes()                     -> entropy-coded footprint (below bits/weight)

Honest: quantization is the lossy step (as it would be anyway); the entropy-coding on top is LOSSLESS — it
recovers the exact quantized values, so a forward pass is byte-identical to the plain-quantized model. Measured
on real gpt2 in `bench_weights.py`. Run: python -m warp_compress.weight_store
"""
from __future__ import annotations

import numpy as np

from .gpu_rrr_wavelet import RRRWaveletGPU


class QuantizedWeightStore:
    """A weight tensor quantized to `bits` and entropy-coded with the RRR wavelet, GPU-addressable."""

    def __init__(self, W, bits: int = 4, device: str = "cuda:0"):
        W = np.asarray(W, np.float32)
        self.shape = W.shape
        self.bits = bits
        self.device = device
        lim = (1 << (bits - 1)) - 1                              # e.g. 7 for int4, 127 for int8
        self.scale = float(np.abs(W).max()) / lim + 1e-12
        q = np.clip(np.round(W.ravel() / self.scale), -lim, lim).astype(np.int64) + lim   # -> [0, 2*lim]
        self._zero = lim
        self.n = int(q.shape[0])
        self.wm = RRRWaveletGPU(q, device=device, bits=bits)     # 2*lim+1 <= 2**bits distinct levels

    def size_bytes(self) -> int:
        return self.wm.index_bytes() + 8                         # entropy-coded values + the scale

    def bits_per_weight(self) -> float:
        return self.size_bytes() * 8 / self.n

    def fetch(self, flat_indices) -> np.ndarray:
        vals = self.wm.access(np.asarray(flat_indices, np.int64))
        return ((vals - self._zero).astype(np.float32)) * self.scale

    def reconstruct(self) -> np.ndarray:
        vals = self.wm.access(np.arange(self.n, dtype=np.int64))
        return ((vals - self._zero).astype(np.float32) * self.scale).reshape(self.shape)


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # Gaussian weights (like a real layer): peaky after quantization -> entropy coder wins
    W = (rng.standard_normal((2048, 512)) / np.sqrt(512)).astype(np.float32)
    print(f"device={dev}   synthetic weight tensor {W.shape} ({W.size:,} weights)\n")
    print(f"  {'quant':>6} {'fixed b/w':>10} {'H0':>6} {'RRR b/w':>9} {'vs fixed':>9}  {'recon MSE':>10}")
    for bits in (4, 8):
        st = QuantizedWeightStore(W, bits=bits, device=dev)
        R = st.reconstruct()
        lim = (1 << (bits - 1)) - 1
        q = np.clip(np.round(W.ravel() / st.scale), -lim, lim) + lim
        _, c = np.unique(q, return_counts=True); p = c / c.sum(); H0 = float(-(p * np.log2(p)).sum())
        # bit-exact vs the plain-quantized dequant?
        exact = np.allclose(R, ((q - lim).astype(np.float32) * st.scale).reshape(W.shape))
        mse = float(np.mean((R - W) ** 2))
        print(f"  int{bits:<3} {float(bits):>10.2f} {H0:>6.2f} {st.bits_per_weight():>9.2f} "
              f"{bits / st.bits_per_weight():>7.2f}× {mse:>11.2e}  {'(lossless vs quant ✓)' if exact else 'FAIL'}")
    print("\n=> entropy-code the QUANTIZED stream, LOSSLESSLY (bit-exact quantized values), with GPU random "
          "access. The win scales with histogram peakiness: a plain Gaussian is mild (int4 ~1.2×), REAL model\n"
          "   weights are far peakier (gpt2 int4 → ~1.8×; see bench_weights.py). int8's small/negative margin is "
          "the RRR class-stream floor (~0.27 b/bit × planes) — the next lever to lift. Compose with quant.")


if __name__ == "__main__":
    _demo()
