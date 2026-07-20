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
                 group_size: "int | None" = None, coder: str = "rrr", block: int = 64, outliers: float = 0.0):
        W = np.asarray(W, np.float32)
        self.shape = W.shape
        self.bits = bits
        self.device = device
        self.coder = coder                                      # "rrr" (search + O(1) RA) | "block" (fast bulk decode)
        lim = (1 << (bits - 1)) - 1                              # e.g. 7 for int4, 127 for int8
        self._zero = lim
        flat = W.ravel()
        self.n = int(flat.shape[0])
        self._out_idx = None
        if outliers > 0.0:                                      # SpQR-style: keep the biggest weights in fp16,
            group_size = None                                   # so they don't blow the int4 scale (per-tensor)
            k = max(1, int(outliers * self.n))
            self._out_idx = np.sort(np.argpartition(np.abs(flat), self.n - k)[self.n - k:]).astype(np.int64)
            self._out_val = flat[self._out_idx].astype(np.float16)
            keep = np.ones(self.n, bool); keep[self._out_idx] = False
            self.scale = float(np.abs(flat[keep]).max()) / lim + 1e-12   # scale from NON-outliers (tighter)
            self._scales = None
            per_val = self.scale
        elif group_size is None:                               # one scale for the whole tensor
            self.scale = float(np.abs(flat).max()) / lim + 1e-12
            self._scales = None
            per_val = self.scale
        else:                                                   # a scale per group of `group_size` values
            g = int(group_size)
            ng = (self.n + g - 1) // g
            padded = np.zeros(ng * g, np.float32); padded[: self.n] = flat
            self._scales = (np.abs(padded.reshape(ng, g)).max(1) / lim + 1e-12).astype(np.float32)
            per_val = np.repeat(self._scales, g)[: self.n]
        self.group_size = group_size
        q = np.clip(np.round(flat / per_val), -lim, lim).astype(np.int64) + lim   # -> [0, 2*lim]
        if self._out_idx is not None:
            q[self._out_idx] = lim                              # outliers -> the zero-point (mode) -> peakier stream
        if coder == "block":                                    # DFloat11-style LUT decode: fast whole-tensor path
            from .gpu_block_huffman import BlockHuffmanArray
            self.wm = BlockHuffmanArray(q, block=block, device=device)
        elif coder == "rans":                                   # near-entropy coder — best for skewed low-bit data
            from .gpu_rans import BlockRANSArray
            self.wm = BlockRANSArray(q, block=block, device=device)
        elif huffman:
            from .gpu_rrr_huffman import RRRWaveletGPUHuff
            self.wm = RRRWaveletGPUHuff(q, device=device, bits=bits)
        else:
            self.wm = RRRWaveletGPU(q, device=device, bits=bits)  # 2*lim+1 <= 2**bits distinct levels

    def _per_val_scale(self):
        if self.group_size is None:
            return self.scale
        return np.repeat(self._scales, self.group_size)[: self.n]

    def _all_values(self):
        return self.wm.decode() if self.coder in ("block", "rans") else self.wm.access(np.arange(self.n, dtype=np.int64))

    def _values_at(self, idx):
        return self.wm.fetch(idx) if self.coder in ("block", "rans") else self.wm.access(idx)

    def size_bytes(self) -> int:
        base = self.wm.size_bits() // 8 if self.coder in ("block", "rans") else self.wm.index_bytes()
        base += self._scales.shape[0] * 2 if self._scales is not None else 8   # fp16 scale side-channel
        if self._out_idx is not None:                          # sparse outliers: int32 index + fp16 value
            base += self._out_idx.shape[0] * 6
        return base

    def bits_per_weight(self) -> float:
        return self.size_bytes() * 8 / self.n

    def fetch(self, flat_indices) -> np.ndarray:
        idx = np.asarray(flat_indices, np.int64)
        vals = self._values_at(idx)
        s = self.scale if self.group_size is None else self._scales[idx // self.group_size]
        out = (vals - self._zero).astype(np.float32) * s
        if self._out_idx is not None:                          # override any requested position that is an outlier
            pos = np.searchsorted(self._out_idx, idx)
            hit = (pos < self._out_idx.shape[0]) & (self._out_idx[np.clip(pos, 0, self._out_idx.shape[0] - 1)] == idx)
            out[hit] = self._out_val[pos[hit]].astype(np.float32)
        return out

    def reconstruct(self) -> np.ndarray:
        vals = self._all_values()
        recon = (vals - self._zero).astype(np.float32) * self._per_val_scale()
        if self._out_idx is not None:
            recon[self._out_idx] = self._out_val.astype(np.float32)   # exact fp16 at the outlier positions
        return recon.reshape(self.shape)

    # --- serialisation: a compressed weight tensor as one portable ChromoFold container blob ---
    def save(self) -> bytes:
        from . import format as fmt
        from .gpu_rrr_huffman import RRRWaveletGPUHuff
        huff = isinstance(self.wm, RRRWaveletGPUHuff)
        wparams, warrays = self.wm.to_host()
        params = {"bits": self.bits, "shape": list(self.shape), "zero": self._zero, "n": self.n,
                  "group_size": self.group_size, "huffman": huff, "coder": self.coder, "wm": wparams}
        if self.group_size is None:
            params["scale"] = self.scale
        else:
            warrays = {**warrays, "_scales": self._scales.astype(np.float32)}
        if self._out_idx is not None:
            params["outliers"] = int(self._out_idx.shape[0])
            warrays = {**warrays, "_out_idx": self._out_idx.astype(np.int64),
                       "_out_val": self._out_val.astype(np.float16)}
        code = self.coder if self.coder in ("block", "rans") else ("huffman" if huff else "rrr")
        config = {"quantize": f"int{self.bits}", "transform": "none",
                  "code": code, "group_size": self.group_size}
        # monotone index metadata (superblocks, word bases, block offsets) delta+zlib-compresses without
        # losing random access; the high-entropy bitstream stays raw.
        monotone = {"sbrank", "sboff", "sbclass", "cbase", "obase", "offbase", "block_off", "byte_off",
                    "_out_idx"} & set(warrays)
        return fmt.pack("weight_store", config, params, warrays, compress=monotone)

    @classmethod
    def load(cls, data: bytes, device: str = "cuda:0"):
        from . import format as fmt
        header, arrays = fmt.unpack(data)
        p = header["params"]
        self = cls.__new__(cls)
        self.shape, self.bits, self._zero, self.n = tuple(p["shape"]), p["bits"], p["zero"], p["n"]
        self.group_size, self.device, self.coder = p["group_size"], device, p.get("coder", "rrr")
        if p["group_size"] is None:
            self.scale, self._scales = p["scale"], None
        else:
            self.scale, self._scales = None, np.asarray(arrays["_scales"], np.float32)
        if p.get("outliers"):
            self._out_idx = np.asarray(arrays["_out_idx"], np.int64)
            self._out_val = np.asarray(arrays["_out_val"], np.float16)
        else:
            self._out_idx = None
        warrays = {k: v for k, v in arrays.items() if k not in ("_scales", "_out_idx", "_out_val")}
        if self.coder == "block":
            from .gpu_block_huffman import BlockHuffmanArray
            self.wm = BlockHuffmanArray.from_host(p["wm"], warrays, device)
        elif self.coder == "rans":
            from .gpu_rans import BlockRANSArray
            self.wm = BlockRANSArray.from_host(p["wm"], warrays, device)
        else:
            from .gpu_rrr_huffman import RRRWaveletGPUHuff
            wm_cls = RRRWaveletGPUHuff if p["huffman"] else RRRWaveletGPU
            self.wm = wm_cls.from_host(p["wm"], warrays, device)
        return self


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    # Gaussian weights (like a real layer) + a few large outliers (real tensors have heavy-tailed channels):
    # peaky after quantization -> entropy coder wins; the outliers blow the int4 scale -> SpQR side-channel wins
    W = (rng.standard_normal((2048, 512)) / np.sqrt(512)).astype(np.float32)
    oi = rng.choice(W.size, int(0.003 * W.size), replace=False)
    W.ravel()[oi] = (rng.standard_normal(oi.size) * (8 + 12 * rng.random(oi.size)) / np.sqrt(512)).astype(np.float32)
    print(f"device={dev}   synthetic weight tensor {W.shape} ({W.size:,} weights, {oi.size} heavy-tailed outliers)\n")
    print(f"  {'config':>22} {'b/weight':>9} {'MSE vs fp32':>12}  lossless-over-quant")
    for label, bits, gs, out in [("int4 per-tensor", 4, None, 0.0), ("int4 group-128", 4, 128, 0.0),
                                 ("int4 + 1% outliers", 4, None, 0.01), ("int3 per-tensor", 3, None, 0.0),
                                 ("int3 + 1% outliers", 3, None, 0.01), ("int8 per-tensor", 8, None, 0.0)]:
        st = QuantizedWeightStore(W, bits=bits, device=dev, huffman=True, group_size=gs, outliers=out)
        R = st.reconstruct()
        mse = float(np.mean((R - W) ** 2))
        idx = rng.integers(0, W.size, 3000)                     # fetch() == reconstruct() -> lossless over quant
        ok = np.allclose(st.fetch(idx), R.ravel()[idx], atol=1e-6)
        print(f"  {label:>22} {st.bits_per_weight():>9.2f} {mse:>12.2e}  {'✓' if ok else 'FAIL'}")
    print("\n=> entropy-code the QUANTIZED stream, LOSSLESSLY (bit-exact vs the chosen quant), with GPU random "
          "access. Two accuracy dials trade against size: `group_size` (per-group scales) and `outliers` (SpQR-\n"
          "   style — keep the biggest weights in fp16 so they don't blow the int4 scale; the rest quantizes finely).\n"
          "   Outliers beat group scaling per bit here (they fix the *cause*), and rescue int3, which per-tensor "
          "cannot use at all. Compose with quant; pick the point your task needs.")


if __name__ == "__main__":
    _demo()
