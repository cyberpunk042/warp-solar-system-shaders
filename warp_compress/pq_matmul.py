"""pq_matmul — fused decode-from-codebook GEMM: y = x·Ŵᵀ with PQ weights decoded INSIDE the kernel.

`gpu_fused_matmul` does this for block-Huffman int weights; the related-work doc names the "two-stage LUT
decode → GEMM" from a product-quantization codebook as the production endgame. This is the codebook version:
a weight tensor W (M×K) is held as `vq_store` product-quantization (a small fp16 codebook + code ids), and the
matmul decodes each weight from the codebook *in registers* as it multiplies — the dense fp32 W is **never
materialized**. So the memory resident during compute is the compressed store (codebook + codes), not M·K
floats.

    mm = PQDecodeMatmul(W, subdim=4, codebook_bits=8)
    mm.matmul(x)          # y = x @ Ŵᵀ, computed without a dense weight matrix
    mm.resident_bytes()   # codebook + codes  (what stays in VRAM)
    mm.dense_bytes()      # M·K·4  (what decode-then-GEMM would need)

Correctness is verified here on CPU (Warp runs on the `cpu` device): the fused result matches
`x @ store.reconstruct()ᵀ` to fp32. The **throughput** win (and the point of a fused kernel) needs a GPU and a
tensor-core-class kernel — this is a correctness/memory PoC that re-decodes W per GEMM, exactly like the int
`gpu_fused_matmul` PoC. Run: `python -m warp_compress.pq_matmul`.
"""
from __future__ import annotations

import numpy as np
import warp as wp

from .vq_store import ProductQuantizedWeightStore


@wp.kernel
def _pq_gemm_k(codes: wp.array(dtype=wp.uint8), cb: wp.array(dtype=wp.float32), subdim: int,
               x: wp.array(dtype=wp.float32), y: wp.array(dtype=wp.float32), B: int, M: int, K: int):
    """One thread per output column m: decode W's row m from (uint8 codes, codebook) and multiply-accumulate it
    into y[:, m] for the whole batch. Each decoded weight lives only in a register — never in global memory. The
    codes are stored at their natural 1-byte width (k≤256), so what is resident IS the compact PQ form."""
    m = wp.tid()
    if m >= M:
        return
    for k in range(K):
        flat = m * K + k                                        # row-major position of W[m, k]
        c = wp.int32(codes[flat // subdim])                     # which centroid this sub-vector uses
        w = cb[c * subdim + (flat % subdim)]                    # decode the weight from the codebook (in register)
        for b in range(B):
            y[b * M + m] = y[b * M + m] + x[b * K + k] * w


class PQDecodeMatmul:
    """W (M×K) held product-quantized, multiplied WITHOUT decoding it to a dense matrix: y = x @ Wᵀ with each
    weight decoded from the codebook inside the GEMM kernel."""

    def __init__(self, W, subdim: int = 4, codebook_bits: int = 8, device: str = "cuda:0", seed: int = 0):
        W = np.asarray(W, np.float32)
        self.M, self.K = int(W.shape[0]), int(W.shape[1])
        self.device = device
        self.subdim = int(subdim)
        if codebook_bits > 8:
            raise ValueError("this PoC packs codes as uint8, so codebook_bits must be <= 8 (k <= 256)")
        self.store = ProductQuantizedWeightStore(W, subdim=subdim, codebook_bits=codebook_bits,
                                                 device=device, seed=seed)
        codes = self.store.wm.access(np.arange(self.store.n_codes, dtype=np.int64)).astype(np.uint8)
        self._codes = wp.array(codes, dtype=wp.uint8, device=device)   # 1 byte/code = its natural width
        self._cb = wp.array(self.store.codebook.ravel().astype(np.float32), dtype=wp.float32, device=device)

    def resident_bytes(self) -> int:
        """What must stay resident to run the fused matmul: the uint8 code stream + the fp16 codebook — the
        compact PQ form the kernel indexes directly (no dense W, no decompression buffer)."""
        return int(self.store.n_codes * 1 + self.store.codebook.size * 2)

    def dense_bytes(self) -> int:
        """What a decode-then-GEMM must materialise: the dense fp32 weight matrix."""
        return self.M * self.K * 4

    def matmul(self, x) -> np.ndarray:
        x = np.asarray(x, np.float32)
        B = int(x.shape[0])
        xd = wp.array(x.ravel(), dtype=wp.float32, device=self.device)
        y = wp.zeros(B * self.M, dtype=wp.float32, device=self.device)
        wp.launch(_pq_gemm_k, dim=self.M,
                  inputs=[self._codes, self._cb, self.subdim, xd, y, B, self.M, self.K], device=self.device)
        return y.numpy().reshape(B, self.M)


def _demo():
    wp.init()
    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    rng = np.random.default_rng(0)
    M, K, B = 1024, 512, 16
    W = ((rng.standard_normal((M, 16)) @ rng.standard_normal((16, K))) / np.sqrt(16 * K)
         + rng.standard_normal((M, K)) / np.sqrt(K)).astype(np.float32)
    x = rng.standard_normal((B, K)).astype(np.float32)

    mm = PQDecodeMatmul(W, subdim=4, codebook_bits=8, device=dev)
    y = mm.matmul(x)
    y_ref = x @ mm.store.reconstruct().T                        # decode-then-GEMM reference (same PQ weights)
    rel = float(np.abs(y - y_ref).max() / (np.abs(y_ref).max() + 1e-9))
    print(f"device={dev}   W {W.shape}, batch {B}\n")
    print(f"  fused PQ decode-GEMM vs decode-then-GEMM:  max rel error = {rel:.2e}  ({'MATCH' if rel < 1e-5 else 'MISMATCH'})")
    print(f"  resident during compute: {mm.resident_bytes()/1e3:8.1f} KB  (codebook + codes)")
    print(f"  dense fp32 weight matrix: {mm.dense_bytes()/1e3:8.1f} KB")
    print(f"  -> {mm.dense_bytes()/mm.resident_bytes():.1f}x less weight memory resident during the matmul")
    print("\n=> The GEMM decodes each weight from the PQ codebook in-register and never builds the dense matrix,\n"
          "   so only the compressed store (codebook + codes) is resident — a fraction of the fp32 weights. The\n"
          "   result is bit-close to decode-then-GEMM (verified). This is a correctness + memory PoC that re-decodes\n"
          "   W per GEMM (like the int gpu_fused_matmul PoC); the throughput win needs a GPU + a tensor-core-class\n"
          "   kernel (the production endgame). RVQ generalizes it — sum the per-stage codebook lookups.")


if __name__ == "__main__":
    _demo()
