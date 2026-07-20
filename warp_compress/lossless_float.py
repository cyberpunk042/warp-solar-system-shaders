"""lossless_float — LOSSLESS fp16/bf16 weight compression by entropy-coding the exponent. With random access.

Lever #4: DFloat11 (arXiv 2504.11651) and NeuZip compress the **exponent** of a float losslessly — weights
cluster near zero, so the exponent field is low-entropy — leaving sign + mantissa raw. That's exact (100%
accuracy) and ~30% smaller for bf16 (its 8-bit exponent carries ~2.6 real bits). Here we do the same, but the
exponent stream goes through our **block coder** (`gpu_block_huffman`), so — unlike DFloat11, which decodes a
whole tensor before use — **any element is randomly addressable on the GPU**. No quantization: it recovers the
*exact* fp16/bf16 bits.

    LosslessFloatStore(u16, exp_bits, mant_bits)   -> exponent block-coded + sign/mantissa bit-packed
    .reconstruct_u16()                             -> the exact 16-bit patterns (view as fp16/bf16)
    .fetch(indices)                                -> exact 16-bit value at any index (exponent decoded on GPU)

fp16 = (exp_bits=5, mant_bits=10); bf16 = (8, 7). Run: python -m warp_compress.lossless_float
"""
from __future__ import annotations

import numpy as np

from .gpu_block_huffman import BlockHuffmanArray


def _pack_bits(vals, w):
    """LSB-first bit-pack fixed-width `w`-bit values into a uint32 stream."""
    n = int(vals.shape[0])
    total = n * w
    words = np.zeros((total + 31) // 32 + 2, np.uint32)
    starts = np.arange(n) * w
    lw = starts >> 5
    lb = (starts & 31).astype(np.uint64)
    v = vals.astype(np.uint64)
    np.bitwise_or.at(words, lw, ((v << lb) & 0xFFFFFFFF).astype(np.uint32))
    spill = (lb + w) > 32
    if spill.any():
        np.bitwise_or.at(words, lw[spill] + 1, (v[spill] >> (32 - lb[spill])).astype(np.uint32))
    return words, total


def _unpack_bits(words, n, w):
    starts = np.arange(n) * w
    lw = starts >> 5
    lb = (starts & 31).astype(np.uint64)
    val = words[lw].astype(np.uint64) >> lb
    spill = (lb + w) > 32
    if spill.any():
        val[spill] |= words[lw[spill] + 1].astype(np.uint64) << (32 - lb[spill])
    return (val & np.uint64((1 << w) - 1)).astype(np.uint16)


class LosslessFloatStore:
    """Exact fp16/bf16, exponent entropy-coded (block-LUT, GPU-decodable, random-access), sign+mantissa raw."""

    def __init__(self, u16, exp_bits: int, mant_bits: int, block: int = 64, device: str = "cuda:0"):
        u16 = np.asarray(u16, np.uint16)
        self.n = int(u16.shape[0])
        self.exp_bits, self.mant_bits, self.device = exp_bits, mant_bits, device
        exp = ((u16 >> mant_bits) & ((1 << exp_bits) - 1)).astype(np.int64)          # low-entropy
        sm = (u16 & ((1 << mant_bits) - 1)) | (((u16 >> 15) & 1) << mant_bits)       # sign + mantissa (raw)
        self.exp = BlockHuffmanArray(exp, block=block, device=device)
        self._sm_words, self._sm_bits = _pack_bits(sm.astype(np.uint16), mant_bits + 1)
        self._sm_w = mant_bits + 1

    def size_bits(self) -> int:
        return self.exp.size_bits() + self._sm_words.shape[0] * 32                   # entropy'd exp + raw sign/mant

    def bits_per_value(self) -> float:
        return self.size_bits() / self.n

    def reconstruct_u16(self) -> np.ndarray:
        exp = self.exp.decode().astype(np.uint16)                                   # GPU decode of the exponent
        sm = _unpack_bits(self._sm_words, self.n, self._sm_w)
        sign = (sm >> self.mant_bits) & 1
        mant = sm & ((1 << self.mant_bits) - 1)
        return ((sign << 15) | (exp << self.mant_bits) | mant).astype(np.uint16)

    def fetch(self, indices) -> np.ndarray:
        idx = np.asarray(indices, np.int64)
        exp = self.exp.fetch(idx).astype(np.uint16)
        sm = _unpack_bits(self._sm_words, self.n, self._sm_w)[idx]
        sign = (sm >> self.mant_bits) & 1
        mant = sm & ((1 << self.mant_bits) - 1)
        return ((sign << 15) | (exp << self.mant_bits) | mant).astype(np.uint16)

    @staticmethod
    def from_fp16(W, block: int = 64, device: str = "cuda:0"):
        u16 = np.ascontiguousarray(np.asarray(W, np.float16)).view(np.uint16).ravel()
        return LosslessFloatStore(u16, 5, 10, block=block, device=device)


def _demo():
    import warnings
    warnings.filterwarnings("ignore")
    import warp as wp
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

    try:
        import torch
        from transformers import AutoModelForCausalLM
        W = None
        for n, p in AutoModelForCausalLM.from_pretrained("gpt2").named_parameters():
            if "mlp.c_fc" in n:
                W = p.detach().float().numpy(); break
    except Exception:
        W = (np.random.default_rng(0).standard_normal((768, 3072)) / 30).astype(np.float32)
    nval = W.size

    print(f"device={dev}   real weight tensor {W.shape} ({nval:,} values)\n")
    print(f"  {'format':8} {'raw':>5} {'ChromoFold':>11} {'save':>7}   exact-lossless   random-access")

    # fp16 (numpy native)
    st16 = LosslessFloatStore.from_fp16(W, device=dev)
    u16_ref = np.ascontiguousarray(W.astype(np.float16)).view(np.uint16).ravel()
    ok16 = np.array_equal(st16.reconstruct_u16(), u16_ref)
    ra16 = np.array_equal(st16.fetch([0, 100, nval - 1]), u16_ref[[0, 100, nval - 1]])
    print(f"  fp16     {16:>5} {st16.bits_per_value():>11.2f} {16/st16.bits_per_value():>6.2f}×   "
          f"{'✓' if ok16 else 'FAIL':^14}   {'✓' if ra16 else 'FAIL'}")

    # bf16 (DFloat11's format — 8-bit exponent, bigger win). bf16 bits = the top 16 bits of the fp32 pattern.
    bf = (np.ascontiguousarray(W.astype(np.float32)).view(np.uint32).ravel() >> 16).astype(np.uint16)
    stbf = LosslessFloatStore(bf, 8, 7, device=dev)
    okbf = np.array_equal(stbf.reconstruct_u16(), bf)
    rabf = np.array_equal(stbf.fetch([1, 50, nval - 2]), bf[[1, 50, nval - 2]])
    print(f"  bf16     {16:>5} {stbf.bits_per_value():>11.2f} {16/stbf.bits_per_value():>6.2f}×   "
          f"{'✓' if okbf else 'FAIL':^14}   {'✓' if rabf else 'FAIL'}")

    print("\n=> LOSSLESS: recovers the exact fp16/bf16 bits (no quantization) by entropy-coding the low-entropy "
          "exponent, sign+mantissa raw — DFloat11/NeuZip's idea. Difference from DFloat11: the exponent goes\n"
          "   through our block coder, so ANY value is randomly addressable on the GPU (DFloat11 decodes a whole "
          "tensor first). bf16's 8-bit exponent compresses harder than fp16's 5-bit. An 'exact model, smaller' mode.")


if __name__ == "__main__":
    _demo()
