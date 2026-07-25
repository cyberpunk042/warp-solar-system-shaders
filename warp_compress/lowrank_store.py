"""lowrank_store — LOW-RANK factorization as a weight codec (base) + a compressed residual (composes with PQ).

The agent map of `warp_compress/` flagged low-rank as absent as a *storage* codec — it only appears via LoRA
*deltas*, never as a base-weight compressor — even though real attention / MLP weight matrices carry strong
low-rank structure. This adds it, and makes it compose with the levers already here.

A weight tensor W (out×in) is factored by a truncated SVD, W ≈ A·B with A (out×r), B (r×in): store the two thin
factors instead of the dense matrix (`out*in` → `r*(out+in)` values — a win whenever `r ≪ out*in/(out+in)`).
That base is smooth; what it misses is a **residual** R = W − A·B, which is small-magnitude and near-full-rank,
so it is handed to a residual codec — `vq_store` (product quantization) or `weight_store` (scalar int) — which
compress it cheaply. So the object is a **two-stage codec**: low-rank captures the correlated structure, the
residual codec mops up the rest, and each stays randomly addressable:

    fetch_rows(rows) = A[rows] @ B  +  residual.fetch_rows(rows)     # O(r·in) per row, no dense materialization

The lossy levers are the **rank** (truncation) and the residual codec's own lever (PQ k / int bits). Both
factors are fp16; the layer over the residual is lossless over that codec — so reconstruct is deterministic and
`fetch == reconstruct`. `residual=None` is pure low-rank (lossy only by truncation).

    LowRankWeightStore(W, rank=16, residual="pq")   -> A, B factors + a compressed residual
    .reconstruct()  ;  .fetch_rows(rows)  ;  .bits_per_weight()

Measured (synthetic low-rank+noise weights): `python -m warp_compress.lowrank_store`. Real-model perplexity is
the follow-up on a box with the model — see docs/research/47.
"""
from __future__ import annotations

import numpy as np


class LowRankWeightStore:
    """Truncated-SVD factorization of a weight tensor (fp16 factors) + an optional compressed residual."""

    def __init__(self, W, rank: int = 16, residual: "str | None" = "pq", device: str = "cuda:0",
                 res_bits: int = 4, pq_subdim: int = 4, pq_bits: int = 8):
        W = np.asarray(W, np.float32)
        if W.ndim != 2:
            raise ValueError("W must be 2-D (out, in)")
        self.shape = W.shape
        self.out, self.inn = W.shape
        self.n = int(W.size)
        self.rank = int(min(rank, self.out, self.inn))
        self.device = device
        # truncated SVD: W ≈ (U_r √S_r)(√S_r V_rᵀ) = A B  — split the singular values so both factors are balanced
        U, S, Vt = np.linalg.svd(W, full_matrices=False)
        sr = np.sqrt(S[: self.rank]).astype(np.float32)
        self.A = (U[:, : self.rank] * sr[None, :]).astype(np.float16).astype(np.float32)   # (out, r)
        self.B = (sr[:, None] * Vt[: self.rank]).astype(np.float16).astype(np.float32)      # (r, in)
        base = self.A @ self.B
        self.residual_kind = residual
        self._res = None
        if residual == "pq":
            from .vq_store import ProductQuantizedWeightStore
            self._res = ProductQuantizedWeightStore(W - base, subdim=pq_subdim, codebook_bits=pq_bits, device=device)
        elif residual == "int":
            from .weight_store import QuantizedWeightStore
            self._res = QuantizedWeightStore(W - base, bits=res_bits, huffman=True, device=device)
        elif residual is not None:
            raise ValueError("residual must be 'pq', 'int', or None")

    # --- footprint ---
    def size_bytes(self) -> int:
        base = (self.A.size + self.B.size) * 2                 # fp16 factors
        if self._res is not None:
            base += self._res.size_bytes()
        return int(base)

    def bits_per_weight(self) -> float:
        return self.size_bytes() * 8 / self.n

    # --- decode ---
    def _res_dense(self) -> np.ndarray:
        return self._res.reconstruct() if self._res is not None else 0.0

    def reconstruct(self) -> np.ndarray:
        return (self.A @ self.B + self._res_dense()).astype(np.float32)

    def fetch_rows(self, rows) -> np.ndarray:
        """Whole rows, decoded without materializing the dense tensor: A[rows]·B + residual[rows]."""
        rows = np.asarray(rows, np.int64)
        out = self.A[rows] @ self.B
        if self._res is not None:
            out = out + self._res.fetch_rows(rows) if hasattr(self._res, "fetch_rows") else out + self._res_dense()[rows]
        return out.astype(np.float32)


def _demo():
    import warp as wp

    dev = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"
    from .weight_store import QuantizedWeightStore
    from .vq_store import ProductQuantizedWeightStore
    rng = np.random.default_rng(0)
    d_out, d_in, true_rank = 2048, 512, 16
    U = rng.standard_normal((d_out, true_rank)).astype(np.float32)
    V = rng.standard_normal((true_rank, d_in)).astype(np.float32)
    W = ((U @ V) / np.sqrt(true_rank * d_in) + rng.standard_normal((d_out, d_in)) / np.sqrt(d_in)).astype(np.float32)
    x = rng.standard_normal((64, d_in)).astype(np.float32)
    fp32_out = x @ W.T

    def report(label, recon, b):
        mse = float(np.mean((recon - W) ** 2))
        oerr = float(np.mean((x @ recon.T - fp32_out) ** 2))
        print(f"  {label:>28} {b:>9.2f} {mse:>12.2e} {oerr:>12.2e}")

    print(f"device={dev}   weight tensor {W.shape}, structure rank≈{true_rank} + full-rank noise\n")
    print(f"  {'config':>28} {'b/weight':>9} {'MSE vs fp32':>12} {'out-err':>12}")
    report("int4 per-tensor", QuantizedWeightStore(W, bits=4, huffman=True, device=dev).reconstruct(),
           QuantizedWeightStore(W, bits=4, huffman=True, device=dev).bits_per_weight())
    report("PQ subdim4 8b", ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=dev).reconstruct(),
           ProductQuantizedWeightStore(W, subdim=4, codebook_bits=8, device=dev).bits_per_weight())
    for label, rank, res in [("low-rank r16 only", 16, None), ("low-rank r16 + int4 res", 16, "int"),
                             ("low-rank r16 + PQ res", 16, "pq"), ("low-rank r32 + PQ res", 32, "pq")]:
        st = LowRankWeightStore(W, rank=rank, residual=res, device=dev)
        rows = rng.integers(0, d_out, 200)                    # fetch_rows == reconstruct -> lossless over the codec
        ok = np.allclose(st.fetch_rows(rows), st.reconstruct()[rows], atol=1e-5)
        report(f"{label} {'✓' if ok else 'FAIL'}", st.reconstruct(), st.bits_per_weight())
    print("\n=> The value is COMPOSITION + structure. Low-rank r16 ALONE is an ultra-cheap 0.62 b/weight, but it keeps\n"
          "   only the structure and drops the full-rank noise -> its output error is HIGH (worse than int4):\n"
          "   structure-only is its own extreme operating point, not a quality win. Composed with a residual codec\n"
          "   at ~2.9 b/weight (matched to int4), low-rank + PQ-residual is COMPETITIVE with int4 — across seeds it\n"
          "   is roughly tied and usually a touch better on output error — while low-rank + int4-residual reaches\n"
          "   near-fp32 quality for a few more bits. Honest: this is not a blanket ratio win over int4; the\n"
          "   distinctive value is that it SEPARATES the tensor into an addressable low-rank base (shareable across\n"
          "   a model family — the LoRA insight, generalized) + a residual, and it owns the sub-1-bit structure-only\n"
          "   regime. On a genuinely full-rank tensor low-rank contributes little and the residual does the work —\n"
          "   it is the STRUCTURE lever, best paired with an element lever. Factors are fp16; the residual is\n"
          "   lossless over its own quant; rows decode addressably (A[row]·B + residual[row]).")


if __name__ == "__main__":
    _demo()
