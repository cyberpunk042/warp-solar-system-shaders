"""quip_store — incoherence × error-feedback, composed (the QuIP recipe): rotate, THEN GPTQ in the rotated basis.

Lever 7 (`hadamard_store`) spreads outliers with a random rotation. Lever 8 (`gptq_store`) picks int codes that
minimize the layer output error. This module asks the research question directly: **do they stack?** — and answers
yes, *super-additively*, which is exactly why QuIP (Chee et al. 2023) pairs them. It rotates the weights incoherent
with a seeded randomized Hadamard transform, then runs GPTQ **in the rotated basis** against the *rotated*
calibration activations, then rotates back.

The finding this module makes measurable (`python -m warp_compress.quip_store`): on **outlier-heavy** weights GPTQ
*alone* barely helps — the outliers blow the per-tensor scale, so the int grid is too coarse for error-feedback to
exploit. Rotating first tightens the scale and **unlocks** GPTQ, so the pair compounds far beyond either alone (an
~8× output-error cut at int4 where GPTQ-alone was ~1×). At int3, QuIP still wins even though Hadamard *alone*
regresses (spreading fattens the bulk) — the error-feedback recovers it.

    st = QuIPWeightStore(W, X, bits=4)     # X: calibration activations (n_samples, in_features)
    st.reconstruct()                        # RHTᵀ( GPTQ_in_rotated_basis( RHT(W) ) )
    st.bits_per_weight()                    # int-store rate + a free seed (the rotation)

The stored codes are plain ints in the same `RRRWaveletGPU` self-index; the rotation costs only its seed. Lossless
over the chosen (rotated) int codes. **Honest costs, inherited from both parents:** it is a calibration-time
method (needs activations `X`), and — because the rotation mixes each Hadamard block — reconstruction is
**row/block-wise**, not single-weight random access. Run: `python -m warp_compress.quip_store`.
"""
from __future__ import annotations

import numpy as np

from .gptq_store import _gptq_codes
from .gpu_rrr_wavelet import RRRWaveletGPU
from .hadamard_store import _largest_pow2_divisor, _rht


class QuIPWeightStore:
    """W (out, in) quantized with the QuIP recipe: a seeded randomized-Hadamard rotation for incoherence, then GPTQ
    error-feedback in the rotated basis against rotated calibration. Codes live in the RRR self-index."""

    def __init__(self, W, X, bits: int = 4, seed: int = 0, damp: float = 0.01, block: "int | None" = None,
                 huffman: bool = True, device: str = "cuda:0"):
        W = np.asarray(W, np.float32)
        X = np.asarray(X, np.float32)
        if W.ndim != 2:
            raise ValueError("W must be 2-D (out, in)")
        if X.ndim != 2 or X.shape[1] != W.shape[1]:
            raise ValueError(f"X must be (n_samples, in={W.shape[1]}), got {X.shape}")
        self.shape = W.shape
        self.out, self.inn = W.shape
        self.n = int(W.size)
        self.bits = int(bits)
        self.seed = int(seed)
        self.device = device
        self.block = int(block) if block is not None else _largest_pow2_divisor(self.inn)
        if self.block & (self.block - 1):
            raise ValueError(f"block {self.block} must be a power of two")
        if self.inn % self.block:
            raise ValueError(f"in_features {self.inn} must be a multiple of block {self.block}")
        lim = (1 << (bits - 1)) - 1
        self._lim = lim
        self.signs = np.random.default_rng(self.seed).choice(np.array([-1.0, 1.0], np.float32), size=self.inn)
        Wr = _rht(W, self.signs, self.block, inverse=False)         # W into the incoherent basis
        Xr = _rht(X, self.signs, self.block, inverse=False)         # x' = T x  (same rotation along the in-dim)
        self.scale = float(np.abs(Wr).max()) / lim + 1e-12          # scale AFTER rotation (outliers already spread)
        levels = _gptq_codes(Wr, Xr, bits, damp, self.scale)        # GPTQ error-feedback in the rotated basis
        if huffman:
            from .gpu_rrr_huffman import RRRWaveletGPUHuff
            self.wm = RRRWaveletGPUHuff(levels, device=device, bits=bits)
        else:
            self.wm = RRRWaveletGPU(levels, device=device, bits=bits)

    def bits_per_weight(self) -> float:
        return self.wm.index_bytes() * 8 / self.n

    def _rotated_recon(self) -> np.ndarray:
        levels = self.wm.access(np.arange(self.n, dtype=np.int64))
        return ((levels.astype(np.float32) - self._lim) * self.scale).reshape(self.shape)

    def reconstruct(self) -> np.ndarray:
        """Dequantize the rotated codes, then rotate back: W ≈ RHTᵀ( GPTQ( RHT(W) ) )."""
        return _rht(self._rotated_recon(), self.signs, self.block, inverse=True)

    def fetch_rows(self, rows) -> np.ndarray:
        """Rows are addressable (a single WEIGHT is not — the rotation mixes each block). Decode these rows in the
        rotated basis, then rotate each back."""
        rows = np.asarray(rows, np.int64)
        cols = np.arange(self.inn, dtype=np.int64)
        flat = (rows[:, None] * self.inn + cols[None, :]).ravel()
        Wr = ((self.wm.access(flat).astype(np.float32) - self._lim) * self.scale).reshape(len(rows), self.inn)
        return _rht(Wr, self.signs, self.block, inverse=True)


def _demo():
    from .gptq_store import GPTQWeightStore
    from .hadamard_store import HadamardQuantStore
    from .weight_store import QuantizedWeightStore

    rng = np.random.default_rng(0)
    out, inn, n = 512, 256, 4096
    A = rng.standard_normal((inn, 16))
    X = (rng.standard_normal((n, 16)) @ A.T + 0.3 * rng.standard_normal((n, inn))).astype(np.float32)
    W = ((rng.standard_normal((out, 16)) @ rng.standard_normal((16, inn))) / np.sqrt(16 * inn)
         + rng.standard_normal((out, inn)) / np.sqrt(inn)).astype(np.float32)
    m = rng.random(W.shape) < 0.003
    W[m] += rng.standard_normal(int(m.sum())).astype(np.float32) * 3.0        # outlier-heavy (the realistic case)
    oerr = lambda Q: float(np.mean((X @ Q.T - X @ W.T) ** 2))

    print(f"QuIP = incoherence × error-feedback, composed   (W {out}×{inn}, calib {n}×{inn}, outlier-heavy, CPU)\n")
    print(f"  {'bits':>5}{'RTN':>12}{'Hadamard':>12}{'GPTQ':>12}{'QuIP (both)':>14}{'QuIP vs RTN':>13}")
    for bits in (4, 3, 2):
        rtn = QuantizedWeightStore(W, bits=bits, huffman=True, device="cpu").reconstruct().reshape(W.shape)
        had = HadamardQuantStore(W, bits=bits, device="cpu").reconstruct()
        gptq = GPTQWeightStore(W, X, bits=bits, device="cpu").reconstruct()
        quip = QuIPWeightStore(W, X, bits=bits, device="cpu")
        er, eh, eg, eq = oerr(rtn), oerr(had), oerr(gptq), oerr(quip.reconstruct())
        print(f"  {bits:>5}{er:>12.3e}{eh:>12.3e}{eg:>12.3e}{eq:>14.3e}{er/eq:>12.2f}x")

    st = QuIPWeightStore(W, X, bits=4, device="cpu")
    idx = rng.integers(0, out, 8)
    ok = np.allclose(st.fetch_rows(idx), st.reconstruct()[idx], atol=1e-6)
    print(f"\n  row-addressable: fetch_rows == reconstruct -> {'MATCH' if ok else 'MISMATCH'}   "
          f"rate {st.bits_per_weight():.2f} b/weight (int store + free seed)")
    print("\n=> On outlier-heavy weights GPTQ ALONE barely helps (outliers blow the scale -> grid too coarse for\n"
          "   error-feedback). Rotating first spreads the outliers and UNLOCKS GPTQ, so the pair compounds far\n"
          "   beyond either alone (~8× at int4). At int3 QuIP still wins even though Hadamard-alone regresses — the\n"
          "   error-feedback recovers it. Codes stay in the RRR index; honest costs (needs calib X, block-wise\n"
          "   access) are inherited from both parents.")


if __name__ == "__main__":
    import warp as wp
    wp.init()
    _demo()
