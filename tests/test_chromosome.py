"""Tests for Process 6 — the chromosome (warp_shaders.genome.chromosome), the honest, chained version.

Process 6 chains from Process 5's ACTUAL output (the 30 nm fibres) and folds them into the two sister
chromatids of the metaphase chromosome — joined at the centromere, capped by the telomeres. Every base
pair is folded (not regenerated) onto its arm; nothing spawned.

  1. chained + conserved: the input IS Process 5's fibres (fib_a/fib_b == coil_fibre output), count same;
  2. the fold is a real chromosome: two sister chromatids on opposite sides, a centromere waist where the
     arms pinch, telomere tints at the tips;
  3. the warp_chromosome scene renders and the whole field animates (fibres -> chromosome).

    python -m tests.test_chromosome
"""

import numpy as np

from warp_shaders.genome import coil_fibre, fold_chromosome


def main():
    fb = coil_fibre(sub=2, block=5)
    cr = fold_chromosome(sub=2, block=5)

    # 1. chained + conserved — the input is exactly Process 5's fibres
    assert cr.n_pairs == fb.n_pairs, f"pairs {cr.n_pairs} != {fb.n_pairs}"
    assert np.array_equal(cr.fib_a, fb.fib_a) and np.array_equal(cr.fib_b, fb.fib_b), \
        "Process 6 does not chain from Process 5's fibres"
    assert np.all(np.isfinite(cr.chr_a)) and np.all(np.isfinite(cr.chr_b))
    print(f"  chained + conserved: OK  ({cr.n_pairs} pairs, input == Process 5)")

    # 2. the fold is a real chromosome
    c0 = cr.chromatid == 0
    c1 = cr.chromatid == 1
    assert c0.sum() > 0 and c1.sum() > 0, "both sister chromatids must be present"
    # sisters sit on opposite sides of the centre
    assert cr.chr_a[c0, 0].mean() < 0.0 < cr.chr_a[c1, 0].mean(), "chromatids not split left/right"
    # centromere waist: near t=0.5 the arms pinch toward the centre (smaller |x| than mid-arm t~0.25)
    waist = np.abs(cr.arm_t - 0.5) < 0.05
    midarm = np.abs(cr.arm_t - 0.25) < 0.05
    assert np.abs(cr.chr_a[waist, 0]).mean() < np.abs(cr.chr_a[midarm, 0]).mean(), \
        "no centromere constriction"
    # telomere caps: the four tips are tinted (bright cyan)
    tip = (cr.arm_t < 0.045) | (cr.arm_t > 0.955)
    assert tip.sum() > 0 and cr.a_col[tip, 1].mean() > 0.85, "telomere tips not tinted"
    # the chromosome is taller than it is wide (the classic upright X)
    assert np.ptp(cr.chr_a[:, 1]) > 1.5 * np.ptp(cr.chr_a[:, 0]), "not an upright chromosome"
    print(f"  real chromosome: OK  (2 chromatids, centromere waist, telomere caps, "
          f"{np.ptp(cr.chr_a[:, 1]):.0f} tall x {np.ptp(cr.chr_a[:, 0]):.0f} wide)")

    # 3. the warp_chromosome scene renders and the whole field animates (fibres -> chromosome)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    fibres = np.asarray(ws.render("warp_chromosome", width=140, height=140, time=0.4), np.float32)
    chromo = np.asarray(ws.render("warp_chromosome", width=140, height=140, time=6.0), np.float32)
    assert np.all(np.isfinite(fibres)) and chromo.max() > 0.1 and chromo.std() > 0.01, "bad frame"
    assert np.abs(fibres - chromo).mean() > 1e-3, "warp_chromosome: fibres -> chromosome did not animate"
    print("  scene warp_chromosome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
