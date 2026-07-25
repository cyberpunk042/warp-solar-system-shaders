"""hadamard_store — incoherence processing: a randomized Hadamard rotation BEFORE quantization (QuIP-style).

Scalar quantization spends its levels on the tensor's *dynamic range*, so a handful of outlier weights blow the
scale and every other weight pays for them (that is why `weight_store` has an fp16 outlier side-channel). The
QuIP / QuIP# line of work removes the outliers a different way: multiply the weight matrix by a **random
orthogonal transform** first. A randomized Hadamard transform (RHT = a random sign flip + a Walsh–Hadamard
transform) is orthogonal, so it preserves the matmul, but it **spreads every outlier across the whole block** —
the rotated matrix is "incoherent" (near-Gaussian, no spikes), which quantizes with far less error at the *same*
bit width. Crucially the transform is a **structured random** map: it is fully determined by its seed, so the
"codebook" costs **zero bytes** — only the integer seed is stored.

    st = HadamardQuantStore(W, bits=4)     # rotate W -> quantize (int4) -> hold rotated codes
    st.reconstruct()                        # dequantize, then rotate back:  W ≈ Hᵀ Q(H W) H  (block-wise)
    st.bits_per_weight()                    # the int store's rate + a negligible seed (signs regenerate from it)

The rotated codes live in the same `RRRWaveletGPU` self-index as every other lever, so they stay GPU-resident and
entropy-coded. **Honest cost of incoherence:** a single weight is no longer *independently* addressable — the
rotation mixes each Hadamard block, so you recover a whole row's block and read the element out. It is therefore
row/block-addressable (like the fused-matmul decode path), not element-random-access. Lossless over the chosen
int quantization of the *rotated* weights.

Measured (`python -m warp_compress.hadamard_store`): on outlier-heavy weights the rotation cuts int4 output/MSE
error ~1.6× vs quantizing directly (more on peakier tensors); on already-Gaussian weights the gain is ~none, and
at int3 it can go *negative* here (spreading also fattens the bulk) — both reported. The transform only helps
where there are outliers to spread. Run: `python -m warp_compress.hadamard_store`.
"""
from __future__ import annotations

import numpy as np

from .weight_store import QuantizedWeightStore


def _fwht(a: np.ndarray) -> np.ndarray:
    """In-place-style fast Walsh–Hadamard transform along the last axis (length must be a power of two).

    Unnormalized: `_fwht(_fwht(x)) == b * x`. Vectorized over all leading axes (every row transformed at once)."""
    a = np.array(a, np.float32)
    orig = a.shape
    b = orig[-1]
    a = a.reshape(-1, b).copy()
    h = 1
    while h < b:
        for i in range(0, b, h * 2):
            x = a[:, i:i + h].copy()
            y = a[:, i + h:i + 2 * h].copy()
            a[:, i:i + h] = x + y
            a[:, i + h:i + 2 * h] = x - y
        h *= 2
    return a.reshape(orig)


def _largest_pow2_divisor(n: int) -> int:
    b = 1
    while (n % (b * 2) == 0):
        b *= 2
    return b


def _rht(W: np.ndarray, signs: np.ndarray, block: int, inverse: bool) -> np.ndarray:
    """Apply the block randomized Hadamard transform along the columns of W (out, in).

    Per block of `block` columns the orthogonal map is T = (1/√block)·H·diag(signs). Forward rotates the weight
    into the incoherent basis; `inverse=True` rotates back. Both are `(1/√block)·FWHT` with the sign flip on the
    weight side — self-inverse because (1/√b·H)² = I and diag(signs)² = I."""
    out, inn = W.shape
    nb = inn // block
    sb = signs.reshape(1, nb, block)
    V = W.reshape(out, nb, block)
    if inverse:
        return (_fwht(V) / np.sqrt(block) * sb).reshape(out, inn).astype(np.float32)
    return (_fwht(V * sb) / np.sqrt(block)).reshape(out, inn).astype(np.float32)


class HadamardQuantStore:
    """W (out, in) rotated by a seeded randomized Hadamard transform, then integer-quantized in the rotated basis.

    The rotation makes the tensor incoherent (outliers spread), so a fixed int width quantizes with less error;
    the transform is regenerated from `seed`, so nothing but the int store + the seed is stored."""

    def __init__(self, W, bits: int = 4, block: "int | None" = None, seed: int = 0, huffman: bool = True,
                 device: str = "cuda:0"):
        W = np.asarray(W, np.float32)
        if W.ndim != 2:
            raise ValueError("W must be 2-D (out, in)")
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
        self.signs = np.random.default_rng(self.seed).choice(
            np.array([-1.0, 1.0], np.float32), size=self.inn)      # structured-random: regenerated from the seed
        Wr = _rht(W, self.signs, self.block, inverse=False)         # into the incoherent basis
        self.store = QuantizedWeightStore(Wr, bits=bits, huffman=huffman, device=device)

    def bits_per_weight(self) -> float:
        """The int store's measured rate; the transform adds only a 4-byte seed (signs regenerate from it), i.e.
        32/n bits per weight — negligible — so we report the store's rate as the effective rate."""
        return self.store.bits_per_weight()

    def reconstruct(self) -> np.ndarray:
        """Dequantize the rotated codes, then rotate back:  W ≈ RHTᵀ( Q( RHT(W) ) )  (exact up to the int step)."""
        Wr = self.store.reconstruct().reshape(self.shape)
        return _rht(Wr, self.signs, self.block, inverse=True)

    def fetch_rows(self, rows) -> np.ndarray:
        """Rows are addressable: dequantize just these rows in the rotated basis, then rotate each back. (A single
        WEIGHT is not independently addressable — the rotation mixes each block — so we return whole rows.)"""
        rows = np.asarray(rows, np.int64)
        cols = np.arange(self.inn, dtype=np.int64)
        flat = (rows[:, None] * self.inn + cols[None, :]).ravel()
        Wr = self.store.fetch(flat).reshape(len(rows), self.inn)
        return _rht(Wr, self.signs, self.block, inverse=True)

    def save(self) -> dict:
        return {"kind": "hadamard", "shape": self.shape, "bits": self.bits, "block": self.block,
                "seed": self.seed, "store": self.store.reconstruct()}   # store the rotated dequant for exact reload

    @classmethod
    def load(cls, blob: dict, device: str = "cpu") -> "HadamardQuantStore":
        # reconstruct is deterministic from (rotated dequant, seed, block); rebuild by re-rotating the saved recon
        signs = np.random.default_rng(int(blob["seed"])).choice(np.array([-1.0, 1.0], np.float32),
                                                                size=blob["shape"][1])
        W = _rht(np.asarray(blob["store"], np.float32), signs, int(blob["block"]), inverse=True)
        return cls(W, bits=int(blob["bits"]), block=int(blob["block"]), seed=int(blob["seed"]), device=device)


def _demo():
    rng = np.random.default_rng(0)
    out, inn = 2048, 512
    x = rng.standard_normal((48, inn)).astype(np.float32)
    oerr = lambda R, W: float(np.mean((x @ R.T - x @ W.T) ** 2))

    def make(outliers: float):
        W = (rng.standard_normal((out, inn)) / np.sqrt(inn)).astype(np.float32)
        if outliers > 0:
            m = rng.random(W.shape) < outliers
            W[m] += rng.standard_normal(int(m.sum())).astype(np.float32) * 3.0   # heavy spikes
        return W

    print(f"randomized-Hadamard incoherence vs direct int quant   (W {out}×{inn}, CPU)\n")
    print(f"  {'weights':<18}{'bits':>5}{'direct MSE':>13}{'hadamard MSE':>14}{'direct oerr':>13}{'had oerr':>11}{'gain':>7}")
    for label, ol in (("outlier-heavy", 0.002), ("near-Gaussian", 0.0)):
        W = make(ol)
        for bits in (4, 3):
            direct = QuantizedWeightStore(W, bits=bits, huffman=True, device="cpu").reconstruct().reshape(W.shape)
            had = HadamardQuantStore(W, bits=bits, device="cpu").reconstruct()
            md, mh = float(((direct - W) ** 2).mean()), float(((had - W) ** 2).mean())
            od, oh = oerr(direct, W), oerr(had, W)
            print(f"  {label:<18}{bits:>5}{md:>13.3e}{mh:>14.3e}{od:>13.3e}{oh:>11.3e}{md/mh:>6.2f}x")

    W = make(0.002)
    st = HadamardQuantStore(W, bits=4, device="cpu")
    idx = rng.integers(0, out, 8)
    ok = np.allclose(st.fetch_rows(idx), st.reconstruct()[idx], atol=1e-6)
    print(f"\n  row-addressable: fetch_rows == full reconstruct  -> {'MATCH' if ok else 'MISMATCH'}")
    print(f"  effective rate: {st.bits_per_weight():.2f} b/weight (int4 store + a 4-byte seed; the transform is free)")
    print("\n=> A seeded randomized Hadamard rotation spreads outliers before quantizing, cutting int4 error ~1.6× on\n"
          "   outlier-heavy weights (near-neutral on Gaussian weights; at int3 it can even go negative — spreading\n"
          "   fattens the bulk too — both reported). The rotated codes stay in the RRR self-index; the transform\n"
          "   costs a seed, not a codebook. Honest trade: incoherence makes access row/block-wise, not per-weight.")


if __name__ == "__main__":
    import warp as wp
    wp.init()
    _demo()
