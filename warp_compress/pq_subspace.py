"""pq_subspace — PROPER product quantization: a separate codebook per column-subspace (Jégou et al. 2011).

`vq_store` learns ONE codebook shared across every sub-vector of the tensor. Standard product quantization
learns a **codebook per subspace**: split the input dimension into blocks of `subdim` columns, and give each
column-block its own k-means codebook, trained only on that block's statistics. Columns of a weight matrix have
different scales and correlations, so a per-block codebook fits each better than one global codebook at the same
code width — the textbook PQ result, and the refinement `vq_store` flagged as future work.

It stays randomly addressable: subspace `s`'s codes are their own token stream in the `RRRWaveletGPU` index, and
a weight `W[o, c]` decodes as `codebook[c // subdim][ code_s[o] , c % subdim ]`. The lossy lever is the (per-
subspace) k-means assignment; the layer on top is lossless over the fp16 codebooks (`fetch == reconstruct`).

    SubspacePQWeightStore(W, subdim=4, codebook_bits=8)   -> in/subdim codebooks + in/subdim code streams
    .reconstruct()  ;  .fetch(flat_indices)  ;  .fetch_rows(rows)  ;  .bits_per_weight()

Honest trade: per-subspace codebooks cost `n_subspaces x` more codebook storage than the shared one (real for a
small tensor, amortized for a big one), buying lower error at the same code width. Measured
(`python -m warp_compress.pq_subspace`). Real-model perplexity is the follow-up — see docs/research/47.
"""
from __future__ import annotations

import numpy as np

from .gpu_rrr_wavelet import RRRWaveletGPU
from .vq_store import _kmeans


class SubspacePQWeightStore:
    """Per-subspace product quantization of W (out, in): one codebook per block of `subdim` columns."""

    def __init__(self, W, subdim: int = 4, codebook_bits: int = 8, device: str = "cuda:0",
                 kmeans_iters: int = 25, seed: int = 0):
        W = np.asarray(W, np.float32)
        if W.ndim != 2:
            raise ValueError("W must be 2-D (out, in)")
        self.shape = W.shape
        self.out, self.inn = W.shape
        self.n = int(W.size)
        self.subdim = int(subdim)
        self.codebook_bits = int(codebook_bits)
        self.device = device
        if self.inn % self.subdim != 0:
            raise ValueError(f"in_features {self.inn} must be a multiple of subdim {self.subdim}")
        self.n_sub = self.inn // self.subdim
        k = 1 << self.codebook_bits
        blocks = W.reshape(self.out, self.n_sub, self.subdim)   # (out, n_sub, subdim)
        self.codebooks = []                                     # n_sub x (<=k, subdim) fp16
        self.wms = []                                           # n_sub code streams, each length `out`
        for s in range(self.n_sub):
            cb, codes = _kmeans(blocks[:, s, :], k, iters=kmeans_iters, seed=seed + s)   # fit THIS column block
            self.codebooks.append(cb.astype(np.float16).astype(np.float32))
            self.wms.append(RRRWaveletGPU(codes.astype(np.int64), device=device, bits=self.codebook_bits))
        self.k = int(self.codebooks[0].shape[0])

    def size_bytes(self) -> int:
        base = sum(wm.index_bytes() for wm in self.wms)          # n_sub entropy-coded code streams
        base += sum(cb.size * 2 for cb in self.codebooks)       # n_sub fp16 codebooks
        return int(base)

    def bits_per_weight(self) -> float:
        return self.size_bytes() * 8 / self.n

    def reconstruct(self) -> np.ndarray:
        out = np.empty((self.out, self.n_sub, self.subdim), np.float32)
        idx = np.arange(self.out, dtype=np.int64)
        for s, (cb, wm) in enumerate(zip(self.codebooks, self.wms)):
            out[:, s, :] = cb[wm.access(idx)]                   # gather this block's centroids per row
        return out.reshape(self.shape).astype(np.float32)

    def fetch(self, flat_indices) -> np.ndarray:
        idx = np.asarray(flat_indices, np.int64)
        row = idx // self.inn
        col = idx % self.inn
        sub = col // self.subdim                                # which subspace (codebook)
        lane = col % self.subdim
        out = np.empty(idx.shape[0], np.float32)
        for s in np.unique(sub):                                # group queries by subspace -> one access per codebook
            m = sub == s
            codes = self.wms[int(s)].access(row[m])
            out[m] = self.codebooks[int(s)][codes, lane[m]]
        return out.astype(np.float32)

    def fetch_rows(self, rows) -> np.ndarray:
        rows = np.asarray(rows, np.int64)
        out = np.empty((rows.shape[0], self.n_sub, self.subdim), np.float32)
        for s, (cb, wm) in enumerate(zip(self.codebooks, self.wms)):
            out[:, s, :] = cb[wm.access(rows)]
        return out.reshape(rows.shape[0], self.inn).astype(np.float32)


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    from .vq_store import ProductQuantizedWeightStore
    rng = np.random.default_rng(0)
    d_out, d_in = 2048, 512
    # columns with DIFFERENT per-block scales — exactly what a per-subspace codebook exploits
    base = (rng.standard_normal((d_out, 16)) @ rng.standard_normal((16, d_in))) / np.sqrt(16 * d_in)
    colscale = (0.2 + 2.0 * rng.random(d_in)).astype(np.float32)          # heterogeneous column magnitudes
    W = (base * colscale[None, :] + rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)
    x = rng.standard_normal((64, d_in)).astype(np.float32)
    fp32_out = x @ W.T

    def report(label, st):
        R = st.reconstruct()
        mse = float(np.mean((R - W) ** 2))
        oerr = float(np.mean((x @ R.T - fp32_out) ** 2))
        print(f"  {label:>28} {st.bits_per_weight():>9.2f} {mse:>12.2e} {oerr:>12.2e}")

    print(f"device={dev}   weight tensor {W.shape}, heterogeneous per-column scales\n")
    print(f"  {'config':>28} {'b/weight':>9} {'MSE vs fp32':>12} {'out-err':>12}")
    for sd, cb in [(4, 8), (4, 6), (8, 8)]:
        shared = ProductQuantizedWeightStore(W, subdim=sd, codebook_bits=cb, device=dev)
        sub = SubspacePQWeightStore(W, subdim=sd, codebook_bits=cb, device=dev)
        idx = rng.integers(0, W.size, 3000)
        ok = np.allclose(sub.fetch(idx), sub.reconstruct().ravel()[idx], atol=1e-6)
        report(f"shared PQ subdim{sd} {cb}b", shared)
        report(f"per-subspace PQ subdim{sd} {cb}b {'✓' if ok else 'FAIL'}", sub)
    print("\n=> A codebook PER column-block fits each block's own statistics, so at the SAME CODE WIDTH it clearly\n"
          "   beats the single shared codebook on error (columns here have heterogeneous scales, as real weight\n"
          "   matrices do). BUT it stores in/subdim codebooks, an overhead of ~k/out bits/weight. On this narrow\n"
          "   tensor (out=2048, k=256 -> +2 b/w) that overhead dominates, so at MATCHED BITS the shared codebook is\n"
          "   actually better (shared 8b at 2.24 b/w beats per-subspace 6b at 2.27). Per-subspace wins at matched\n"
          "   bits only when the layer is WIDE enough to amortize the codebooks (out >> k) — the regime of real LLM\n"
          "   projection matrices. Honest lever: better fit per code-bit, a per-out-row codebook tax. Codes stay in\n"
          "   the RRR index (addressable); lossy lever = per-subspace k-means; lossless on top (fetch == recon).")


if __name__ == "__main__":
    _demo()
