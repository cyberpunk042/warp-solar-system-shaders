"""Tests for Process 6 — the metaphase chromosome (warp_shaders.genome.chromosome).

Operator spec: the 30nm fibre folds into the blue X — the whole ladder lands on the chromosome. A
conserving process: every base pair (every bit of the card) packed into the chromosome body once.

  1. conservation: one point per base pair (== tokens/2), fibre and chromo both present;
  2. shape: the chromo fills a four-armed X (points reach all four (±x,±y) quadrants), pinched at
     the centromere (few points near the origin), spread in a plane (thin in z);
  3. condensation: the chromosome body is far more compact than the fibre it folded from;
  4. the warp_chromosome scene renders and animates (gathered fibre -> folded X).

    python -m tests.test_chromosome
"""

import numpy as np

from warp_shaders.genome import bind_pairs, fold_chromosome


def main():
    n_pairs = bind_pairs(sub=2, block=5).n_pairs
    ch = fold_chromosome(sub=2, block=5)
    c = ch.chromo

    # 1. conservation
    assert ch.n_pairs == n_pairs, f"points {ch.n_pairs} != base pairs {n_pairs}"
    assert c.shape == (n_pairs, 3) and np.all(np.isfinite(c)) and np.all(np.isfinite(ch.fiber))
    print(f"  conservation: OK  ({ch.n_pairs} base pairs packed into the chromosome, none spawned)")

    # 2. shape — four arms populated, centromere pinched, planar (thin in z)
    q = [(c[:, 0] > 0.25) & (c[:, 1] > 0.4), (c[:, 0] < -0.25) & (c[:, 1] > 0.4),
         (c[:, 0] > 0.25) & (c[:, 1] < -0.4), (c[:, 0] < -0.25) & (c[:, 1] < -0.4)]
    counts = [int(m.sum()) for m in q]
    assert all(k > n_pairs // 14 for k in counts), f"four arms not populated: {counts}"
    near_centre = int((np.abs(c[:, 1]) < 0.15).sum())
    assert near_centre < n_pairs // 6, "centromere not pinched"
    assert c[:, 2].std() < c[:, 1].std(), "chromosome not planar (z should be thin)"
    print(f"  shape: OK  (four arms {counts}, pinched centromere, planar)")

    # 3. condensation — the X is far more compact than the fibre
    fib_extent = ch.fiber[:, 0].max() - ch.fiber[:, 0].min()
    chr_extent = np.linalg.norm(c.max(0) - c.min(0))
    assert chr_extent < 0.2 * fib_extent, f"not condensed ({chr_extent:.1f} vs fibre {fib_extent:.0f})"
    print(f"  condensation: OK  (fibre {fib_extent:.0f} -> chromosome {chr_extent:.1f} across)")

    # 4. the warp_chromosome scene renders and animates (gathered fibre -> folded X)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    a = np.asarray(ws.render("warp_chromosome", width=120, height=120, time=0.0), np.float32)
    x = np.asarray(ws.render("warp_chromosome", width=120, height=120, time=3.6), np.float32)
    assert np.all(np.isfinite(a)) and x.max() > 0.1 and x.std() > 0.01, "bad frame"
    assert np.abs(a - x).mean() > 1e-3, "warp_chromosome: fibre -> X did not animate"
    print("  scene warp_chromosome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
