"""bench_super_elastic — the elastic depth dial: stacking layers vs one bigger fold.

The operator's "super-elastic" question, measured: does adding *more layers of matrix operation* squeeze
the fold further than spending the same effort on one bigger single-layer fold? The honest answer has two
faces, both printed here:

  * at MATCHED total rank, splitting one rank-R fold into L smaller layers is ~a wash (each layer re-pays
    a per-matrix scale; the factorization is no tighter) — more layers is NOT free lunch.
  * on the ACCURACY axis it is decisive: stacking re-quantizes each layer's leftover with a fresh scale,
    so error falls toward zero (to bit-exact with a final residual), while a single low-rank layer
    PLATEAUS — adding rank past the true structural rank cannot fix the quantization-error floor.

So super-elastic's value is the **elastic accuracy dial**: spend layers to buy accuracy the single fold
can't reach, all the way to lossless. Pure numpy+zlib. Run: python -m warp_compress.bench_super_elastic
"""
from __future__ import annotations

import numpy as np

from . import grouped_delta as gd
from . import super_elastic as se


def _rank4_group(rng, n=16, h=64, w=64, amp=8.0):
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    b = rng.standard_normal((n, 4))
    v = rng.standard_normal((4, h * w))
    return np.clip(base.reshape(-1) + amp * (b @ v), 0, 255).astype(np.uint8).reshape(n, h, w)


def main():
    rng = np.random.default_rng(20260725)
    g = _rank4_group(rng)
    base = se.baseline_bytes(g)
    print("=== super-elastic: the elastic depth dial (rank-4 group, 16x64x64 uint8) ===\n")

    print("A. accuracy axis -- STACK rank-4 layers: error falls toward zero (the elastic squeeze):")
    print(f"   {'layers':>7} {'atom_bytes':>11} {'ratio':>7} {'mean_abs_err':>13}")
    prev = None
    for L in (1, 2, 3, 4, 6):
        x = se.compress(g, layers=L, rank=4)
        e = se.mean_abs_err(g, x)
        b = se.encoded_bytes(x)
        assert prev is None or e <= prev + 1e-9, "error must not rise with depth"
        prev = e
        print(f"   {L:>7} {b:>11} {base / b:>6.2f}x {e:>13.4f}")

    print("\nB. the single-fold alternative -- add rank instead: it PLATEAUS, never reaching 0:")
    print(f"   {'rank':>7} {'atom_bytes':>11} {'ratio':>7} {'mean_abs_err':>13}")
    single = {}
    for r in (4, 8, 12, 16):
        y = gd.compress(g, rank=r, quant_factors=True)
        single[r] = gd.mean_abs_err(g, y)
        print(f"   {r:>7} {gd.encoded_bytes(y):>11} {base / gd.encoded_bytes(y):>6.2f}x {single[r]:>13.4f}")

    se4 = se.mean_abs_err(g, se.compress(g, layers=4, rank=4))
    print(f"\n   -> super-elastic L=4 reaches err {se4:.4f}; the single fold plateaus at "
          f"~{single[16]:.4f} (rank 16). Stacking buys accuracy rank cannot.")

    print("\nC. lossless -- a final exact residual makes the whole stack bit-exact + still compressed:")
    rep = se.compress_group(g, layers=3, rank=4, final_residual=True)
    print(f"   L=3 rank=4 +final_residual: lossless={rep['lossless']}  atom={rep['atom_bytes']}B  "
          f"ratio_vs_baseline={rep['ratio_vs_baseline']:.2f}x")

    # honest internal sanity
    assert se4 < single[16], "super-elastic must beat the single-fold error plateau"
    assert rep["lossless"] and rep["ratio_vs_baseline"] > 1.0
    print("\nSummary: more layers != free lunch at matched rank, but the elastic dial buys accuracy a "
          "single low-rank fold cannot -- down to bit-exact -- measured, not assumed.")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
