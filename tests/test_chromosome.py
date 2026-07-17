"""Tests for Process 7 — the chromosome (warp_shaders.genome.chromosome), single-chromatid form.

Process 7 chains from Process 6's ACTUAL output (the telomere-capped fibre) and folds it into a single
condensed chromatid — a rod with a centromere constriction and the two real telomere t-loops capping its
ends. One continuous strand, nothing copied — fully conserving.

  1. chained + conserved: the input IS Process 6's capped strand (tel_a/tel_b == cap_telomeres), count same;
  2. a real chromatid: an upright rod, a centromere waist (thinner than the arms), the two telomere caps
     carried to the tips;
  3. the warp_chromosome scene renders and the whole field animates (capped strand -> chromatid).

    python -m tests.test_chromosome
"""

import numpy as np

from warp_shaders.genome import fold_chromosome
from warp_shaders.genome.telomere import cap_telomeres


def main():
    tl = cap_telomeres(sub=2, block=5)
    cr = fold_chromosome(sub=2, block=5)

    # 1. chained + conserved — the input is exactly Process 6's telomere-capped strand
    assert cr.n_pairs == tl.n_pairs, f"pairs {cr.n_pairs} != {tl.n_pairs}"
    assert np.array_equal(cr.tel_a, tl.tel_a) and np.array_equal(cr.tel_b, tl.tel_b), \
        "Process 7 does not chain from Process 6's capped strand"
    assert np.all(np.isfinite(cr.chr_a)) and np.all(np.isfinite(cr.chr_b))
    print(f"  chained + conserved: OK  ({cr.n_pairs} pairs, input == Process 6)")

    # 2. a real single chromatid
    s = cr.arm_s
    r = np.hypot(cr.chr_a[:, 0], cr.chr_a[:, 2])          # radius from the rod axis
    arm = r[np.abs(s - 0.25) < 0.03].mean()
    waist = r[np.abs(s - 0.5) < 0.03].mean()
    assert waist < 0.75 * arm, f"no centromere constriction (waist {waist:.2f} vs arm {arm:.2f})"
    # the two telomeres are carried to the tips (extreme |y|) and there are exactly two
    assert int(cr.is_tel.sum()) == 2 * tl.tel_len, "telomere count wrong"
    tel_y = np.abs(cr.chr_a[cr.is_tel, 1]).mean()
    body_y = np.abs(cr.chr_a[~cr.is_tel, 1]).mean()
    assert tel_y > body_y, "telomere caps not carried to the tips"
    # upright rod (taller than wide)
    assert np.ptp(cr.chr_a[:, 1]) > 1.6 * np.ptp(cr.chr_a[:, 0]), "not an upright chromatid"
    print(f"  single chromatid: OK  (centromere waist {waist:.2f} < arm {arm:.2f}, 2 telomere caps, "
          f"{np.ptp(cr.chr_a[:, 1]):.0f} tall x {np.ptp(cr.chr_a[:, 0]):.0f} wide)")

    # 3. the warp_chromosome scene renders and the whole field animates (capped strand -> chromatid)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    strand = np.asarray(ws.render("warp_chromosome", width=130, height=150, time=0.4), np.float32)
    chromo = np.asarray(ws.render("warp_chromosome", width=130, height=150, time=6.0), np.float32)
    assert np.all(np.isfinite(strand)) and chromo.max() > 0.1 and chromo.std() > 0.01, "bad frame"
    assert np.abs(strand - chromo).mean() > 1e-3, "warp_chromosome: strand -> chromatid did not animate"
    print("  scene warp_chromosome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
