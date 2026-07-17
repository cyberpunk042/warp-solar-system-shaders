"""Tests for Process 8 — replication → the metaphase X (warp_shaders.genome.replication).

Process 8 chains from Process 7's single chromatid and replicates it into two identical sister chromatids,
tilted into the metaphase X and joined at the centromere. Replication is the one deliberate copy (S-phase),
shown not hidden; everything else is conserving.

  1. chained + a real copy: the pre-replication state IS two copies of Process 7's chromatid; the sisters
     are identical copies;
  2. the X is real: the two sisters tilt apart (wider than the single chromatid), four telomere caps in all;
  3. the warp_chromosome_x scene renders and the whole field animates (chromatid -> X).

    python -m tests.test_replication
"""

import numpy as np

from warp_shaders.genome import fold_chromosome
from warp_shaders.genome.replication import replicate_chromosome


def main():
    cr = fold_chromosome(sub=2, block=5)
    rp = replicate_chromosome(sub=2, block=5)
    p = cr.n_pairs

    # 1. chained + a real copy — the pre-replication state is two copies of Process 7's chromatid
    assert rp.n_pairs == p, f"pairs {rp.n_pairs} != {p}"
    assert rp.single_a.shape[0] == 2 * p, "replication should carry two sisters (2 * n_pairs)"
    assert np.array_equal(rp.single_a[:p], cr.chr_a), "sister A is not Process 7's chromatid"
    assert np.array_equal(rp.single_a[p:], cr.chr_a), "sister B is not an identical copy"
    assert np.all(np.isfinite(rp.x_a)) and np.all(np.isfinite(rp.x_b))
    print(f"  chained + identical copy: OK  ({p} pairs, two sisters, sister B == sister A)")

    # 2. the X is real — the sisters tilt apart, so the pair is wider than one chromatid; four telomeres
    single_w = float(np.ptp(rp.single_a[:, 0]))
    x_w = float(np.ptp(rp.x_a[:, 0]))
    assert x_w > 1.3 * single_w, f"sisters did not splay into an X ({x_w:.1f} vs single {single_w:.1f})"
    assert int(rp.is_tel.sum()) == 2 * int(cr.is_tel.sum()), "the X should have four telomeres, not two"
    print(f"  metaphase X: OK  (splay {x_w:.0f} > single {single_w:.0f}, four telomere caps)")

    # 3. the warp_chromosome_x scene renders and the whole field animates (chromatid -> X)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    one = np.asarray(ws.render("warp_chromosome_x", width=130, height=150, time=0.4), np.float32)
    ex = np.asarray(ws.render("warp_chromosome_x", width=130, height=150, time=6.0), np.float32)
    assert np.all(np.isfinite(one)) and ex.max() > 0.1 and ex.std() > 0.01, "bad frame"
    assert np.abs(one - ex).mean() > 1e-3, "warp_chromosome_x: chromatid -> X did not animate"
    print("  scene warp_chromosome_x: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
