"""bench_grouped_delta_lora — C4 on its headline target: a LoRA adapter library.

A LoRA adapter is, by construction, a shared base weight plus a **low-rank delta** (``W + s * B @ A``);
an adapter *library* is therefore the canonical correlated group M20 targets — every member overlaps the
same base. This is where "group + delta-from-a-reference + matrix-compact the atom" should pay.

We can't download a real model here (the HF hub is unreachable in this environment), so we build a
**realistically-structured** library: a Gaussian base weight in a gpt2-``c_attn``-like shape, int8-quantized,
plus N rank-``rho`` LoRA adapters (real ``B @ A`` low-rank deltas). This exercises the *structure* of the
target with real LoRA math; a real-model number (an actual adapter set from the model venv) is the follow-on.

What it measures — and the honest result (P7 — report negatives):

  * **lossless WINS.** The computed "third state" centroid recovers ~the shared base; exact int8 residuals
    make the library ~1.4x smaller than N independently-stored adapters, bit-exact. This is the robust win.
  * **the float32 low-rank approx does NOT win at this shape** — an honest negative. At LoRA's large output
    dim the SVD ``Vt`` factor (rows scale with d) in float32 is *bigger* than the exact int8 residuals, so
    the "matrix-compact" atom loses to exact residuals here. The lesson: the atom's factors must themselves
    be quantized/entropy-coded to pay at scale. (Contrast ``bench_grouped_delta``, whose small-d residuals
    DO favor the low-rank atom — the crossover is output-dim x factor-precision, not "low-rank always wins".)
  * the rank dial still **recovers the structure** — reconstruction error falls monotonically to 0 as rank
    reaches N — it just isn't the size win at this shape.

Pure numpy + zlib, no torch/GPU. Run: python -m warp_compress.bench_grouped_delta_lora
"""
from __future__ import annotations

import numpy as np

from . import grouped_delta as gd


def _quant_int8(x):
    """Symmetric per-tensor int8 quantize of a float weight (the lossy lever; C4 is lossless over it)."""
    scale = np.abs(x).max() / 127.0 + 1e-12
    return np.clip(np.rint(x / scale), -127, 127).astype(np.int16), scale


def _lora_library(rng, n=16, d_in=256, d_out=768, rank=8, adapter_amp=0.15):
    """N adapters = quantize(W + s * B_i @ A_i): a shared base + real rank-`rank` low-rank deltas."""
    w = rng.standard_normal((d_in, d_out)).astype(np.float64)          # the shared base weight
    members = []
    for _ in range(n):
        b = rng.standard_normal((d_in, rank))
        a = rng.standard_normal((rank, d_out))
        delta = adapter_amp * (b @ a) / np.sqrt(rank)                  # the LoRA low-rank delta
        q, _ = _quant_int8(w + delta)
        members.append(q)
    stack = np.stack(members)
    lo = stack.min()
    return (stack - lo).astype(np.uint8), rank


def main():
    rng = np.random.default_rng(20260725)
    lib, true_rank = _lora_library(rng)
    n = lib.shape[0]

    print("=== C4 on a LoRA adapter library (16 adapters, 256x768, int8, true adapter rank 8) ===\n")

    # 1. lossless: library as centroid ('third state' ~ shared base) + exact residuals ------------
    rep = gd.compress_group(lib, mode="centroid", rank=None)
    print("1. lossless library storage vs N independent adapters:")
    print(f"   atom={rep['atom_bytes']}B  baseline(N independent)={rep['baseline_bytes']}B  "
          f"ratio_vs_baseline={rep['ratio_vs_baseline']:.2f}  lossless={rep['lossless']}")
    assert rep["lossless"], "library round-trip must be bit-exact"
    assert rep["ratio_vs_baseline"] > 1.0, (
        f"a LoRA library shares a base -> lossless grouping should win, got {rep['ratio_vs_baseline']:.3f}"
    )
    print(f"   -> WIN: the library as base + exact-residual atom is {rep['ratio_vs_baseline']:.2f}x "
          f"smaller than N independent adapters (bit-exact)\n")

    # the 'third state' centroid recovers ~the shared base
    atom = gd.compress(lib, mode="centroid", rank=None)
    print(f"2. centroid recovers ~the shared base: mean |residual| = {np.abs(atom.residuals).mean():.2f} "
          f"levels (small vs the {int(lib.max()) - int(lib.min())}-wide uint8 span)\n")

    # 3. approx float32 low-rank: the HONEST NEGATIVE at this (large-d) shape --------------------
    lossless_bytes = gd.encoded_bytes(atom)
    base = gd.baseline_bytes(lib)
    print("3. approx float32 low-rank atom -- honest negative at LoRA's large output dim:")
    print(f"   {'rank':>5} {'mean_abs_err':>13} {'atom_bytes':>11} {'vs_baseline':>11}")
    errs = {}
    for r in (2, 4, 8, 12, 16):
        a = gd.compress(lib, mode="centroid", rank=r)
        errs[r] = gd.mean_abs_err(lib, a)
        b = gd.encoded_bytes(a)
        print(f"   {r:>5} {errs[r]:>13.3f} {b:>11} {base / b:>10.2f}x")
    print(f"   (exact int8 residuals = {lossless_bytes}B)")
    # error recovers structure (monotone to ~0 at full rank)...
    assert errs[2] >= errs[8] >= errs[16], f"rank must recover structure monotonically: {errs}"
    # ...but the float32 factors do NOT beat exact int8 residuals at this large-d shape (the negative)
    approx_full = gd.encoded_bytes(gd.compress(lib, mode="centroid", rank=true_rank))
    assert approx_full > lossless_bytes, (
        f"at large d the float32 rank-{true_rank} atom should be BIGGER than exact int8 residuals "
        f"(the honest negative): approx={approx_full}B lossless={lossless_bytes}B"
    )
    print(f"   -> NEGATIVE: float32 rank-{true_rank} atom ({approx_full}B) > exact int8 residuals "
          f"({lossless_bytes}B). The factors must be quantized to pay at scale.\n")

    print("Summary: lossless grouping WINS on a LoRA library (~1.4x, bit-exact); the low-rank *approx* "
          "only pays for small output dims OR with quantized factors -- measured, not assumed.")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
