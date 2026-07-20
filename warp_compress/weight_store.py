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
    """A weight tensor quantized to `bits` and entropy-coded with the RRR wavelet, GPU-addressable. With
    ``huffman=True`` the class stream is Huffman-coded (near the plane H0 — best for the very skewed low bits
    of quantized weights)."""

    def __init__(self, W, bits: int = 4, device: str = "cuda:0", huffman: bool = False,
                 group_size: "int | None" = None):
        W = np.asarray(W, np.float32)
        self.shape = W.shape
        self.bits = bits
        self.device = device
        self.group_size = group_size
        lim = (1 << (bits - 1)) - 1                              # e.g. 7 for int4, 127 for int8
        self._zero = lim
        flat = W.ravel()
        self.n = int(flat.shape[0])
        if group_size is None:                                  # one scale for the whole tensor
            self.scale = float(np.abs(flat).max()) / lim + 1e-12
            self._scales = None
            per_val = self.scale
        else:                                                   # a scale per group of `group_size` values
            g = int(group_size)
            ng = (self.n + g - 1) // g
            padded = np.zeros(ng * g, np.float32); padded[: self.n] = flat
            self._scales = (np.abs(padded.reshape(ng, g)).max(1) / lim + 1e-12).astype(np.float32)
            per_val = np.repeat(self._scales, g)[: self.n]
        q = np.clip(np.round(flat / per_val), -lim, lim).astype(np.int64) + lim   # -> [0, 2*lim]
        if huffman:
            from .gpu_rrr_huffman import RRRWaveletGPUHuff
            self.wm = RRRWaveletGPUHuff(q, device=device, bits=bits)
        else:
            self.wm = RRRWaveletGPU(q, device=device, bits=bits)  # 2*lim+1 <= 2**bits distinct levels

    def _per_val_scale(self):
        if self.group_size is None:
            return self.scale
        return np.repeat(self._scales, self.group_size)[: self.n]

    def size_bytes(self) -> int:
        base = self.wm.index_bytes()
        return base + (self._scales.shape[0] * 2 if self._scales is not None else 8)   # fp16 scale side-channel

    def bits_per_weight(self) -> float:
        return self.size_bytes() * 8 / self.n

    def fetch(self, flat_indices) -> np.ndarray:
        idx = np.asarray(flat_indices, np.int64)
        vals = self.wm.access(idx)
        s = self.scale if self.group_size is None else self._scales[idx // self.group_size]
        return (vals - self._zero).astype(np.float32) * s

    def reconstruct(self) -> np.ndarray:
        vals = self.wm.access(np.arange(self.n, dtype=np.int64))
        return ((vals - self._zero).astype(np.float32) * self._per_val_scale()).reshape(self.shape)

    # --- serialisation: a compressed weight tensor as one portable ChromoFold container blob ---
    def save(self) -> bytes:
        from . import format as fmt
        from .gpu_rrr_huffman import RRRWaveletGPUHuff
        huff = isinstance(self.wm, RRRWaveletGPUHuff)
        wparams, warrays = self.wm.to_host()
        params = {"bits": self.bits, "shape": list(self.shape), "zero": self._zero, "n": self.n,
                  "group_size": self.group_size, "huffman": huff, "wm": wparams}
        if self.group_size is None:
            params["scale"] = self.scale
        else:
            warrays = {**warrays, "_scales": self._scales.astype(np.float32)}
        config = {"quantize": f"int{self.bits}", "transform": "none",
                  "code": "huffman" if huff else "rrr", "group_size": self.group_size}
        return fmt.pack("weight_store", config, params, warrays)

    @classmethod
    def load(cls, data: bytes, device: str = "cuda:0"):
        from . import format as fmt
        from .gpu_rrr_huffman import RRRWaveletGPUHuff
        header, arrays = fmt.unpack(data)
        p = header["params"]
        self = cls.__new__(cls)
        self.shape, self.bits, self._zero, self.n = tuple(p["shape"]), p["bits"], p["zero"], p["n"]
        self.group_size, self.device = p["group_size"], device
        if p["group_size"] is None:
            self.scale, self._scales = p["scale"], None
        else:
            self.scale, self._scales = None, np.asarray(arrays["_scales"], np.float32)
        wm_cls = RRRWaveletGPUHuff if p["huffman"] else RRRWaveletGPU
        warrays = {k: v for k, v in arrays.items() if k != "_scales"}
        self.wm = wm_cls.from_host(p["wm"], warrays, device)
        return self


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # Gaussian weights (like a real layer): peaky after quantization -> entropy coder wins
    W = (rng.standard_normal((2048, 512)) / np.sqrt(512)).astype(np.float32)
    print(f"device={dev}   synthetic weight tensor {W.shape} ({W.size:,} weights)\n")
    print(f"  {'config':>18} {'b/weight':>9} {'MSE vs fp32':>12}  lossless")
    for label, bits, gs in [("int4 per-tensor", 4, None), ("int4 group-128", 4, 128),
                            ("int4 group-32", 4, 32), ("int8 per-tensor", 8, None)]:
        st = QuantizedWeightStore(W, bits=bits, device=dev, huffman=True, group_size=gs)
        R = st.reconstruct()
        mse = float(np.mean((R - W) ** 2))
        lim = (1 << (bits - 1)) - 1
        pv = st._per_val_scale()
        refq = (np.clip(np.round(W.ravel() / pv), -lim, lim) * pv).reshape(W.shape)
        ok = np.allclose(R, refq, atol=1e-5)
        print(f"  {label:>18} {st.bits_per_weight():>9.2f} {mse:>12.2e}  {'✓' if ok else 'FAIL'}")
    print("\n=> entropy-code the QUANTIZED stream, LOSSLESSLY (bit-exact vs the chosen quant), with GPU random "
          "access. The `group_size` knob is the accuracy↔size Pareto dial: per-tensor int4 is tiny but coarse;\n"
          "   group-128 int4 gets ~int8 accuracy at ~0.7× int8's size (per-group scales trade the peaky-histogram "
          "compression for accuracy). Compose with quant; pick the point your task needs.")


if __name__ == "__main__":
    _demo()
