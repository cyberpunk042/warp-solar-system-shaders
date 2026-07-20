"""gpu_fused_matmul — decode the compressed weights INSIDE the matmul (lever #7, the decode-during-compute endgame).

Every other path decodes weights to a dense tensor first, then multiplies. This fuses the two: a GEMM kernel that
reads the block-Huffman-coded int4 weights straight from the compressed stream and decodes each weight on the fly
as it multiplies — so the dequantized (M×K) matrix is **never materialised in VRAM**. Only the compressed store
is resident during the compute. That is the literal "fit a bigger model on one GPU *while computing with it*"
claim, and it's what Marlin does for fixed-width int4 (arXiv/IST-DASLab) — here for a *variable-length entropy*
code, which the fixed-COUNT block layout makes tractable: the bitstream is contiguous, so a thread starts at any
row's bit offset (`block_off`) and decodes that row inline.

    FusedDecodeMatmul(W, bits=4, block=64)   -> quantize + block-Huffman-code W (M×K), resident compressed
    .matmul(x)                               -> y = x @ Wᵀ, weights decoded in-kernel (no dense W)
    .reference(x)                            -> decode-to-dense then GEMM (same result, materialises W)
    .resident_bytes() / .dense_bytes()       -> the memory the fused path saves during compute

Honest: fusing re-decodes W per GEMM, so it trades compute for memory (the ChromoFold thesis) — a Warp
proof-of-concept, not a tensor-core Marlin kernel. Lossless over the quantized values. Run:
python -m warp_compress.gpu_fused_matmul
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .weight_store import QuantizedWeightStore

wp.init()


@wp.func
def _decode_at(words: wp.array(dtype=wp.uint32), lut: wp.array(dtype=wp.int32), maxlen: int, pos: int) -> int:
    """Table-decode one canonical-Huffman code at bit position `pos`; returns symbol | (code_length << 8)."""
    look = int(0)
    for k in range(maxlen):
        wpos = pos + k
        look = (look << 1) | int((words[wpos >> 5] >> wp.uint32(31 - (wpos & 31))) & wp.uint32(1))
    return lut[look]


@wp.kernel
def _fused_matmul_k(words: wp.array(dtype=wp.uint32), block_off: wp.array(dtype=wp.int32),
                    lut: wp.array(dtype=wp.int32), maxlen: int, block: int,
                    x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32),
                    B: int, M: int, K: int, scale: float, zero: int):
    """One thread per output column m: decode W's row m from the compressed stream and multiply-accumulate it
    into y[:, m] for the whole batch. The dequantized row lives only in registers, never in global memory."""
    m = wp.tid()
    if m >= M:
        return
    start = m * K                                             # flat index of W[m, 0] (row-major)
    bs = start // block
    pos = block_off[bs]
    skip = start - bs * block
    for _ in range(skip):                                     # advance to the row start (mid-block)
        sl = _decode_at(words, lut, maxlen, pos)
        pos = pos + (sl >> 8)
    for k in range(K):
        sl = _decode_at(words, lut, maxlen, pos)
        q = sl & 0xFF
        pos = pos + (sl >> 8)
        w = float(q - zero) * scale                          # dequantized weight, in a register
        for b in range(B):
            y[b * M + m] = y[b * M + m] + x[b * K + k] * w


class FusedDecodeMatmul:
    """A weight tensor W (M×K) held block-Huffman-compressed, multiplied WITHOUT ever decoding it to a dense
    matrix: y = x @ Wᵀ with each weight decoded inside the GEMM kernel."""

    def __init__(self, W, bits: int = 4, block: int = 64, device: str = "cuda:0"):
        W = np.asarray(W, np.float32)
        self.M, self.K = int(W.shape[0]), int(W.shape[1])
        self.device = device
        self.store = QuantizedWeightStore(W, bits=bits, device=device, coder="block", block=block)
        self.block = int(block)

    def resident_bytes(self) -> int:
        """What must stay in VRAM to run the fused matmul: only the compressed store."""
        return self.store.size_bytes()

    def dense_bytes(self) -> int:
        """What a decode-then-GEMM must materialise: the dequantized weight matrix (fp32)."""
        return self.M * self.K * 4

    def matmul(self, x) -> np.ndarray:
        x = np.asarray(x, np.float32)
        B = int(x.shape[0])
        xd = wp.array(x.ravel(), dtype=wp.float32, device=self.device)
        y = wp.zeros(B * self.M, dtype=wp.float32, device=self.device)
        bh = self.store.wm
        wp.launch(_fused_matmul_k, dim=self.M,
                  inputs=[bh.words, bh.block_off, bh.lut, bh.maxlen, self.block, xd, y,
                          B, self.M, self.K, float(self.store.scale), int(self.store._zero)],
                  device=self.device)
        wp.synchronize_device(self.device)
        return y.numpy().reshape(B, self.M)

    def reference(self, x) -> np.ndarray:
        """Decode W to dense, then a plain GEMM — same result, but materialises the (M×K) matrix."""
        return np.asarray(x, np.float32) @ self.store.reconstruct().T


def _demo():
    import time

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    M, K, B = 2048, 2048, 8
    W = (rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    x = (rng.standard_normal((B, K)) / np.sqrt(K)).astype(np.float32)

    fm = FusedDecodeMatmul(W, bits=4, block=64, device=dev)
    y_fused = fm.matmul(x)
    y_ref = fm.reference(x)
    err = float(np.max(np.abs(y_fused - y_ref)))
    rel = err / (float(np.max(np.abs(y_ref))) + 1e-12)

    for _ in range(3):
        fm.matmul(x)
    t0 = time.perf_counter(); fm.matmul(x); t_fused = time.perf_counter() - t0

    print(f"device={dev}   W {M}×{K} int4 block-Huffman,  x {B}×{K}\n")
    print(f"[correct] fused (decode-in-GEMM) vs decode-then-dense:  max|Δ|={err:.2e}  rel={rel:.2e}  "
          f"{'✓ lossless-over-quant' if rel < 1e-5 else 'FAIL'}")
    print(f"[memory]  resident during compute:  fused {fm.resident_bytes()/1e6:5.2f} MB (compressed)  vs  "
          f"dense {fm.dense_bytes()/1e6:6.2f} MB (dequantized W)")
    print(f"          => {fm.dense_bytes()/fm.resident_bytes():.1f}× less VRAM held while multiplying "
          f"(never materialise W)")
    print(f"[speed]   fused GEMM {t_fused*1e3:.1f} ms  ({B*M*K/t_fused/1e9:.1f} GFLOP/s) — re-decodes W per "
          f"matmul: the compute-for-memory trade")
    print("\n=> lever #7: the weights are decoded INSIDE the matmul, so the dequantized matrix never exists in "
          "VRAM — only the compressed store is resident during compute. Fixed-count blocks make the variable-\n"
          "   length entropy code fused-decodable (a thread starts at any row's bit offset). Honest: this Warp "
          "kernel trades compute (re-decode per GEMM) for memory; a tensor-core Marlin-class fusion is the endgame.")


if __name__ == "__main__":
    _demo()
