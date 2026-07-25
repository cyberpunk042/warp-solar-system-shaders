"""bench_model_fold — the workload proof (P10): fold a whole 'model', state the model-level squeeze.

The operator's original goal was "fit the most inside GPU memory." M20 (grouped_delta) and M21
(super_elastic) each measured one tensor group; this bench composes them on a **whole-model workload** and
reports the honest model-level statement -- the constitution's decisive-proof shape ("a workload, not a
ratio"). Two regimes, because the fold pays very differently:

  * **independent dense layers** (a normal model's distinct weight matrices) -- the group members are NOT
    correlated, so grouping is ~neutral (the honest negative the repo already knows). Super-elastic still
    gives the accuracy dial: lossless re-encode down to bit-exact, no free capacity from grouping.
  * **a correlated bank** (MoE experts / a LoRA adapter library -- same-shape members that share a base) --
    grouping WINS: the shared reference + folded residuals fit meaningfully more in the same bytes.

So the model-level answer is workload-shaped: the fold buys capacity exactly where the model has
*repeated / adapted* structure, and is entropy-neutral on the independent bulk. Measured, not assumed --
a real model (a true expert bank from the venv) is the SAIN/venv-gated follow-on. Pure numpy+zlib.
Run: python -m warp_compress.bench_model_fold
"""
from __future__ import annotations

import numpy as np

from . import grouped_delta as gd
from . import super_elastic as se


def _independent_layers(rng, n=24, h=64, w=64):
    """n distinct dense weight matrices (a normal model's layers): mutually uncorrelated, int8."""
    return rng.integers(0, 256, size=(n, h, w), dtype=np.uint8)


def _correlated_bank(rng, n=24, h=64, w=64, rank=6, amp=8.0):
    """n same-role members sharing a base + a low-rank spread (an expert bank / adapter library), int8."""
    base = rng.integers(0, 256, size=(h, w), dtype=np.int64)
    b = rng.standard_normal((n, rank))
    v = rng.standard_normal((rank, h * w))
    return np.clip(base.reshape(-1) + amp * (b @ v), 0, 255).astype(np.uint8).reshape(n, h, w)


def _report(name, group):
    native = group.size  # int8 -> 1 byte/elem, the resident footprint before entropy coding
    base = gd.baseline_bytes(group)                                   # each member entropy-coded alone
    lossless = gd.encoded_bytes(gd.compress(group, rank=None))        # M20 lossless (exact residuals)
    # M21 elastic dial: a lossy operating point (few layers) + the bit-exact point (final residual)
    lossy = se.compress_group(group, layers=2, rank=rank_for(group), final_residual=False)
    exact = se.compress_group(group, layers=3, rank=rank_for(group), final_residual=True)
    print(f"  {name}:")
    print(f"    native int8            {native:>9} B")
    print(f"    independent (baseline) {base:>9} B  ({native / base:.2f}x)")
    print(f"    M20 lossless fold      {lossless:>9} B  ({native / lossless:.2f}x vs native, "
          f"{base / lossless:.2f}x vs baseline)")
    print(f"    M21 lossy  (L2)        {lossy['atom_bytes']:>9} B  ({base / lossy['atom_bytes']:.2f}x vs "
          f"baseline, err {lossy['mean_abs_err']:.3f})")
    print(f"    M21 exact  (L3+resid)  {exact['atom_bytes']:>9} B  ({base / exact['atom_bytes']:.2f}x vs "
          f"baseline, lossless={exact['lossless']})")
    return base, lossless


def rank_for(group):
    return min(6, group.shape[0])


def main():
    rng = np.random.default_rng(20260725)
    print("=== model-fold workload (P10): 24 layers of 64x64 int8 weights ===\n")

    ind = _independent_layers(rng)
    cor = _correlated_bank(rng)

    ib, il = _report("independent dense layers", ind)
    print()
    cb, cl = _report("correlated bank (experts / adapters)", cor)

    ind_gain = ib / il
    cor_gain = cb / cl
    print(f"\n  model-level statement (lossless):")
    print(f"    independent bulk -> {ind_gain:.2f}x vs baseline (grouping HURTS: residual-vs-centroid is "
          f"as big as the originals + a reference to store -> keep the bulk on the plain per-tensor coder)")
    print(f"    correlated banks -> {cor_gain:.2f}x vs baseline (grouping fits more in the same VRAM)")
    print("    => the fold is a per-GROUP opt-in: apply it to the model's REPEATED/ADAPTED structure "
          "(experts/adapters/tied layers), skip it on the independent bulk -- measured, honest, "
          "workload-shaped. That selection policy is the model-level win.")

    # honest sanity: correlated wins, independent does NOT (the fold must be applied selectively)
    assert cor_gain > 1.0, f"correlated bank fold must beat baseline: {cor_gain:.2f}x"
    assert ind_gain < 1.0, f"grouping the independent bulk should NOT pay (the negative): {ind_gain:.2f}x"
    assert cor_gain > ind_gain, "correlated must gain more than independent"
    assert np.array_equal(gd.decompress(gd.compress(ind, rank=None)), ind), "independent lossless bit-exact"
    assert np.array_equal(gd.decompress(gd.compress(cor, rank=None)), cor), "correlated lossless bit-exact"
    print("\nALL PASSED")


if __name__ == "__main__":
    main()
