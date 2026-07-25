"""bench_grouped_delta — the Processor-vs-Memory sweet spot for grouped delta superposition (C4).

The operator's second question, measured: *"trying to find the sweet spot for how much to rely on
Processor VS Memory."* This sweeps the C4 transform (:mod:`warp_compress.grouped_delta`) across three
axes and prints the honest envelope so the knee is visible, not asserted:

  1. **correlation break-even** — at what per-member noise does grouping stop beating independent
     storage (``ratio_vs_baseline`` crosses 1.0)? This is the "is this group correlated enough" line.
  2. **group-size amortization** — a bigger correlated group shares one reference across more members,
     so the win should grow with N.
  3. **rank dial (P1)** — the lossy-approx knee: as rank ``r`` rises, reconstruction error falls and the
     atom grows. The sweet spot is the smallest ``r`` whose error is within budget.

Pure numpy + zlib — no torch, no GPU. This is the synthetic-but-structured envelope; a real-model number
(a Qwen2.5 expert bank / adapter set) is the follow-on that needs the model venv.

Run: python -m warp_compress.bench_grouped_delta
"""
from __future__ import annotations

import numpy as np

from . import grouped_delta as gd


def _group(rng, n, h, w, noise):
    """n uint8 tensors = a shared base + uniform per-member noise in [-noise, noise]."""
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    if noise == 0:
        return np.stack([np.clip(base, 0, 255)] * n).astype(np.uint8)
    out = [np.clip(base + rng.integers(-noise, noise + 1, size=(h, w), dtype=np.int64), 0, 255)
           for _ in range(n)]
    return np.stack(out).astype(np.uint8)


def _low_rank_group(rng, n, h, w, true_rank, amp):
    """n tensors whose residuals share `true_rank` directions — for the rank-dial sweep."""
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    dirs = rng.integers(-1, 2, size=(true_rank, h, w), dtype=np.int64)
    out = []
    for _ in range(n):
        c = rng.integers(-amp, amp + 1, size=true_rank, dtype=np.int64)
        resid = sum(int(c[k]) * dirs[k] for k in range(true_rank))
        out.append(np.clip(base + resid, 0, 255))
    return np.stack(out).astype(np.uint8)


def main():
    rng = np.random.default_rng(20260725)
    h = w = 48

    print("=== C4 grouped delta superposition — measured envelope (seed 20260725, 48x48 uint8) ===\n")

    # 1. correlation break-even -----------------------------------------------------------------
    print("1. correlation break-even (n=8, centroid, lossless) — ratio_vs_baseline vs per-member noise:")
    print(f"   {'noise±':>7} {'ratio_vs_baseline':>18} {'verdict':>10}")
    break_even = None
    for noise in (0, 1, 2, 4, 8, 16, 32, 64):
        rep = gd.compress_group(_group(rng, 8, h, w, noise), mode="centroid", rank=None)
        r = rep["ratio_vs_baseline"]
        verdict = "WIN" if r > 1.0 else "lose"
        if break_even is None and r <= 1.0:
            break_even = noise
        print(f"   {noise:>7} {r:>18.3f} {verdict:>10}")
    print(f"   -> grouping stops winning around noise ~{break_even} "
          f"(the 'correlated enough?' line)\n")

    # 2. group-size amortization ----------------------------------------------------------------
    print("2. group-size amortization (noise±2, centroid, lossless) — win grows with N:")
    print(f"   {'N':>4} {'ratio_vs_baseline':>18}")
    prev = None
    grew = True
    for n in (2, 4, 8, 16, 32):
        rep = gd.compress_group(_group(rng, n, h, w, 2), mode="centroid", rank=None)
        r = rep["ratio_vs_baseline"]
        if prev is not None and r < prev - 0.05:
            grew = False
        prev = r
        print(f"   {n:>4} {r:>18.3f}")
    print(f"   -> {'win grows (or holds) with group size' if grew else 'non-monotone'}\n")

    # 3. rank dial (Processor vs Memory) --------------------------------------------------------
    lr = _low_rank_group(rng, n=16, h=h, w=w, true_rank=4, amp=8)
    lossless_bytes = gd.encoded_bytes(gd.compress(lr, mode="centroid", rank=None))
    print("3. rank dial — the Processor-vs-Memory knee (n=16, residuals are true-rank-4):")
    print(f"   {'rank':>5} {'mean_abs_err':>13} {'atom_bytes':>11} {'vs_lossless':>12}")
    print(f"   {'exact':>5} {0.0:>13.3f} {lossless_bytes:>11} {'1.00x':>12}")
    knee = None
    for r in (1, 2, 3, 4, 6, 8):
        atom = gd.compress(lr, mode="centroid", rank=r)
        err = gd.mean_abs_err(lr, atom)
        b = gd.encoded_bytes(atom)
        if knee is None and err <= 0.5:            # within a half-level error budget
            knee = r
        print(f"   {r:>5} {err:>13.3f} {b:>11} {b / lossless_bytes:>11.2f}x")
    print(f"   -> sweet spot ~rank {knee}: first rank within a 0.5-level error budget "
          f"(true structural rank was 4)\n")

    # honest internal sanity (so running the bench also validates the shape of the result)
    assert gd.compress_group(_group(rng, 8, h, w, 1), mode="centroid")["ratio_vs_baseline"] > 1.0
    assert gd.compress_group(_group(rng, 8, h, w, 64), mode="centroid")["ratio_vs_baseline"] <= 1.05
    assert knee is not None and knee <= 4
    print("ALL PASSED")


if __name__ == "__main__":
    main()
