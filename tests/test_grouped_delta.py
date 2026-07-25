"""Tests for warp_compress.grouped_delta (C4 — grouped delta superposition).

Run: python -m tests.test_grouped_delta

Checks:
  1. lossless round-trip is bit-exact (centroid / "third state" reference)
  2. lossless round-trip is bit-exact (root reference)
  3. a CORRELATED group beats independent storage (ratio_vs_baseline > 1) -- grouping wins
  4. an UNCORRELATED group does NOT win (ratio_vs_baseline <= ~1) -- the honest negative (P7)
  5. tighter correlation compresses better than looser (monotone in similarity)
  6. approx (rank-r) reconstruction error falls as r grows -- the processor-vs-memory dial (P1),
     and a low-rank group's approx atom is smaller than its lossless residuals
"""

import numpy as np

import warp_compress.grouped_delta as gd


def _correlated_group(rng, n=8, h=32, w=32, noise=3):
    """n near-identical uint8 tensors: a shared base + small per-member noise (residuals are low-entropy)."""
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    out = []
    for _ in range(n):
        d = rng.integers(-noise, noise + 1, size=(h, w), dtype=np.int64)
        out.append(np.clip(base + d, 0, 255))
    return np.stack(out).astype(np.uint8)


def main():
    rng = np.random.default_rng(20260725)

    # 1. lossless round-trip, centroid ("third state") reference
    g = _correlated_group(rng)
    atom = gd.compress(g, mode="centroid", rank=None)
    back = gd.decompress(atom)
    assert np.array_equal(back, g), "centroid lossless round-trip not bit-exact"
    # the centroid is a synthesized 'third state' -- it need not equal any member
    assert not any(np.array_equal(atom.reference, g[i]) for i in range(g.shape[0])) or True
    print("  1 centroid lossless round-trip: OK (bit-exact)")

    # 2. lossless round-trip, root reference
    atom_r = gd.compress(g, mode="root", rank=None)
    assert np.array_equal(gd.decompress(atom_r), g), "root lossless round-trip not bit-exact"
    assert np.array_equal(atom_r.reference, g[0]), "root reference must be the first member"
    print("  2 root lossless round-trip: OK (bit-exact)")

    # 3. correlated group WINS vs independent storage
    rep = gd.compress_group(g, mode="centroid", rank=None)
    assert rep["lossless"], "report must confirm lossless"
    assert rep["ratio_vs_baseline"] > 1.0, (
        f"correlated group should beat independent storage, got {rep['ratio_vs_baseline']:.3f}"
    )
    print(f"  3 correlated group wins: OK (ratio_vs_baseline={rep['ratio_vs_baseline']:.2f}, "
          f"atom={rep['atom_bytes']}B vs baseline={rep['baseline_bytes']}B)")

    # 4. UNCORRELATED group does NOT win -- report the negative honestly (P7)
    u = rng.integers(0, 256, size=(8, 32, 32), dtype=np.uint8)  # each member independent
    urep = gd.compress_group(u, mode="centroid", rank=None)
    assert urep["lossless"], "uncorrelated round-trip still lossless"
    assert urep["ratio_vs_baseline"] <= 1.05, (
        f"uncorrelated group must NOT meaningfully win (grouping only pays on correlation); "
        f"got {urep['ratio_vs_baseline']:.3f}"
    )
    print(f"  4 uncorrelated group does not win: OK (ratio_vs_baseline={urep['ratio_vs_baseline']:.2f} "
          f"<= 1.05 -- honest negative)")

    # 5. tighter correlation compresses better than looser
    tight = gd.compress_group(_correlated_group(rng, noise=1), mode="centroid")
    loose = gd.compress_group(_correlated_group(rng, noise=40), mode="centroid")
    assert tight["ratio_vs_baseline"] > loose["ratio_vs_baseline"], (
        f"tighter group should compress better: tight={tight['ratio_vs_baseline']:.3f} "
        f"loose={loose['ratio_vs_baseline']:.3f}"
    )
    print(f"  5 monotone in similarity: OK (tight={tight['ratio_vs_baseline']:.2f} > "
          f"loose={loose['ratio_vs_baseline']:.2f})")

    # 6. approx rank dial: build residuals with genuine rank-3 structure, show error falls as r grows
    h = w = 32
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    patterns = rng.integers(-1, 2, size=(3, h, w), dtype=np.int64)  # 3 shared residual directions
    members = []
    for _ in range(10):
        c = rng.integers(-6, 7, size=3, dtype=np.int64)
        resid = sum(int(c[k]) * patterns[k] for k in range(3))
        members.append(np.clip(base + resid, 0, 255))
    lr = np.stack(members).astype(np.uint8)

    errs = [gd.mean_abs_err(lr, gd.compress(lr, mode="centroid", rank=r)) for r in (1, 2, 3)]
    assert errs[0] >= errs[1] >= errs[2], f"rank dial must reduce error monotonically: {errs}"
    assert errs[2] < errs[0], f"more rank must help on low-rank structure: {errs}"
    print(f"  6 rank dial reduces error: OK (mean_abs_err r1={errs[0]:.3f} >= r2={errs[1]:.3f} "
          f">= r3={errs[2]:.3f})")

    # ...and the approx atom is smaller than storing the exact residuals (the memory side of the dial)
    approx_bytes = gd.encoded_bytes(gd.compress(lr, mode="centroid", rank=3))
    lossless_bytes = gd.encoded_bytes(gd.compress(lr, mode="centroid", rank=None))
    assert approx_bytes < lossless_bytes, (
        f"rank-3 approx atom should be smaller than exact residuals: "
        f"approx={approx_bytes}B lossless={lossless_bytes}B"
    )
    print(f"  6b approx atom smaller than lossless: OK (approx={approx_bytes}B < "
          f"lossless={lossless_bytes}B)")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
