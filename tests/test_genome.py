"""Tests for the continuous genome-compression master animation (warp_shaders.scenes.warp_genome).

The master is NOT a hand-rolled coil and NOT a lerp between arbitrary poses — it is the engine's six genome
library processes run back-to-back, each chaining from the previous one's ACTUAL output. So the invariants
are about that real chain (no cheat, matter conserved) and the honest final fold:

  1. the same 182 872 base pairs run through every stage (conserved count);
  2. each stage's END state equals the next stage's START state exactly — the chain is continuous, nothing
     teleports between incompatible representations (field->helix->nucleosome->fibre->telomere->chromatid);
  3. the Process-7 fold is a real compression that starts exactly at the telomere state (chr@f=0 == telomere)
     and ends far more compact (the chromatid extent is a fraction of the fibre band's);
  4. the warp_genome scene renders and animates the whole ladder (extended field -> packed chromatid).

    python -m tests.test_genome
"""

import numpy as np


def main():
    from warp_shaders.genome import (wind_helix, wound_positions, wrap_nucleosomes,
                                      coil_fibre, cap_telomeres)
    from warp_shaders.genome.chromatid import fold_chromatid

    # 1. the conserved thread — the real base-pair count runs through every stage
    hx = wind_helix(sub=2, block=5)
    pa, pb = wound_positions(hx)
    nc = wrap_nucleosomes(sub=2, block=5)
    fb = coil_fibre(sub=2, block=5)
    tl = cap_telomeres(sub=2, block=5)
    ch = fold_chromatid(sub=2, block=5)
    N = 182872
    for name, obj in [("helix", hx), ("nucleosome", nc), ("fibre", fb), ("telomere", tl), ("chromatid", ch)]:
        assert obj.n_pairs == N, f"{name} has {obj.n_pairs} pairs, expected {N}"
    print(f"  conserved thread: OK  ({N} base pairs through all 6 stages)")

    # 2. each stage's end == the next stage's start (the real chain — no teleport, no cheat)
    assert np.allclose(pa, nc.helix_a) and np.allclose(pb, nc.helix_b), "helix end != nucleosome start"
    assert np.allclose(nc.nuc_a, fb.bead_a) and np.allclose(nc.nuc_b, fb.bead_b), "nuc end != fibre start"
    assert np.allclose(fb.fib_a, tl.fib_a) and np.allclose(fb.fib_b, tl.fib_b), "fibre end != telomere start"
    assert np.allclose(tl.tel_a, ch.tel_a) and np.allclose(tl.tel_b, ch.tel_b), "telomere end != fold start"
    print("  chained continuity: OK  (each stage end == next stage start, matter conserved)")

    # 3. the fold: starts exactly at the telomere state and compresses hard into the chromatid
    assert np.all(np.isfinite(ch.chr_a)) and np.all(np.isfinite(ch.chr_b)), "non-finite chromatid"

    def extent(p):
        return float(max(p[:, d].max() - p[:, d].min() for d in range(3)))

    fibre_ext, chr_ext = extent(tl.tel_a), extent(ch.chr_a)
    assert chr_ext < 0.35 * fibre_ext, f"fold not compact enough: {fibre_ext:.1f} -> {chr_ext:.1f}"
    # a real centromere waist: the radius near the middle is smaller than on the arms
    u = np.arange(N) / (N - 1.0)
    r = np.sqrt(ch.chr_a[:, 0] ** 2 + ch.chr_a[:, 2] ** 2)
    waist = r[np.abs(u - 0.5) < 0.02].mean()
    arm = r[np.abs(u - 0.25) < 0.02].mean()
    assert waist < arm, f"no centromere waist: waist {waist:.2f} vs arm {arm:.2f}"
    print(f"  honest fold: OK  (band {fibre_ext:.0f} -> chromatid {chr_ext:.0f}, waist {waist:.1f} < arm {arm:.1f})")

    # 4. the scene renders and animates the whole ladder
    import warp as wp
    import warp_shaders as ws
    wp.init()
    from warp_shaders.scenes import warp_genome as G
    assert G._N == N, f"scene reports {G._N} pairs"
    start = np.asarray(ws.render("warp_genome", width=150, height=120, time=0.4), np.float32)   # base pairs
    end = np.asarray(ws.render("warp_genome", width=150, height=120, time=23.9), np.float32)    # chromatid
    assert np.all(np.isfinite(start)) and end.max() > 0.1 and end.std() > 0.01, "bad frame"
    assert np.abs(start - end).mean() > 1e-3, "warp_genome: the compression did not animate"
    print("  scene warp_genome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
