"""Tests for warp_compress.super_elastic (C5 — super-elastic multi-layer residual fold).

Run: python -m tests.test_super_elastic

Checks:
  1. more layers drive reconstruction error DOWN (the elastic squeeze) — monotone in depth
  2. stacking reaches accuracy a single low-rank layer CANNOT (residual-quant beats the rank plateau)
  3. final_residual=True round-trips BIT-EXACT (the lossless product)
  4. a lossless super-elastic stack still compresses a correlated group vs independent storage
  5. decode = reference + sum of layers (shape + dtype preserved)
"""
import numpy as np

import warp_compress.grouped_delta as gd
import warp_compress.super_elastic as se


def _rank4_group(rng, n=16, h=64, w=64, amp=8.0):
    """A genuinely rank-4 correlated group: shared base + a rank-4 low-rank spread, int8-quantized."""
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    b = rng.standard_normal((n, 4))
    v = rng.standard_normal((4, h * w))
    return np.clip(base.reshape(-1) + amp * (b @ v), 0, 255).astype(np.uint8).reshape(n, h, w)


def main():
    rng = np.random.default_rng(20260725)
    g = _rank4_group(rng)

    # 1. more layers -> lower error (the elastic squeeze)
    errs = [se.mean_abs_err(g, se.compress(g, layers=L, rank=4)) for L in (1, 2, 3, 4)]
    assert errs[0] > errs[1] > errs[2] > errs[3], f"more layers must reduce error: {errs}"
    print(f"  1 elastic squeeze: OK (err L1={errs[0]:.3f} > L2={errs[1]:.3f} > L3={errs[2]:.3f} "
          f"> L4={errs[3]:.3f})")

    # 2. stacking reaches accuracy a single low-rank layer cannot (beats the rank plateau)
    se_err = se.mean_abs_err(g, se.compress(g, layers=4, rank=4))
    single_hi = gd.mean_abs_err(g, gd.compress(g, rank=16, quant_factors=True))
    assert se_err < single_hi, (
        f"super-elastic (L=4 r=4) should beat a single rank-16 layer's error floor: "
        f"se={se_err:.4f} single_r16={single_hi:.4f}"
    )
    print(f"  2 beats the rank plateau: OK (super-elastic L4={se_err:.4f} < single rank-16={single_hi:.4f})")

    # 3. final_residual -> bit-exact lossless
    rep = se.compress_group(g, layers=3, rank=4, final_residual=True)
    assert rep["lossless"], "final_residual must give a bit-exact round-trip"
    x = se.compress(g, layers=3, rank=4, final_residual=True)
    assert np.array_equal(se.decompress(x), g), "lossless round-trip not bit-exact"
    print(f"  3 lossless final_residual: OK (bit-exact; atom={rep['atom_bytes']}B)")

    # 4. a lossless super-elastic stack still compresses a correlated group vs independent storage
    assert rep["ratio_vs_baseline"] > 1.0, (
        f"lossless super-elastic should still beat independent storage on a correlated group, "
        f"got {rep['ratio_vs_baseline']:.3f}"
    )
    print(f"  4 lossless still compresses: OK (ratio_vs_baseline={rep['ratio_vs_baseline']:.2f})")

    # 5. decode shape/dtype
    back = se.decompress(se.compress(g, layers=2, rank=4))
    assert back.shape == g.shape and back.dtype == g.dtype, "decode must preserve shape + dtype"
    print("  5 decode preserves shape + dtype: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
