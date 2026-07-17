"""Tests for Process 4 — nucleosomes (warp_shaders.genome.nucleosome).

Operator spec: coil the double helix into beads on a string — nucleosomes made only of the DNA wound
tighter, nothing spawned. A conserving process.

  1. conservation: one point per base pair (== tokens/2), extended and wrapped both present;
  2. packing: ~200 bp per bead -> the strand condenses (wrapped shorter end-to-end than extended);
  3. beads: wrapped points sit near their bead centres (the wrap is a compact coil, not a spread line);
  4. the warp_nucleosome scene renders and the strand animates (extended -> beads on a string).

    python -m tests.test_nucleosome
"""

import numpy as np

from warp_shaders.genome import bind_pairs, wrap_nucleosomes


def main():
    n_pairs = bind_pairs(sub=2, block=5).n_pairs
    nc = wrap_nucleosomes(sub=2, block=5)

    # 1. conservation — one point per base pair, both forms present
    assert nc.n_pairs == n_pairs, f"points {nc.n_pairs} != base pairs {n_pairs}"
    assert nc.extended.shape == (n_pairs, 3) and nc.wrapped.shape == (n_pairs, 3)
    assert np.all(np.isfinite(nc.extended)) and np.all(np.isfinite(nc.wrapped))
    assert nc.n_beads == (n_pairs + 199) // 200
    print(f"  conservation: OK  ({nc.n_pairs} points -> {nc.n_beads} beads, none spawned)")

    # 2. packing — wrapping condenses the strand end-to-end
    ext_len = nc.extended[:, 0].max() - nc.extended[:, 0].min()
    wr_len = nc.wrapped[:, 0].max() - nc.wrapped[:, 0].min()
    assert wr_len < ext_len, f"wrapping did not condense ({wr_len:.1f} vs {ext_len:.1f})"
    print(f"  packing: OK  (strand condenses {ext_len:.0f} -> {wr_len:.0f} along its length)")

    # 3. beads — the first bead's wrapped points cluster tightly (a compact coil, radius < 1)
    first = nc.wrapped[:146]
    spread = np.linalg.norm(first - first.mean(0), axis=1).max()
    assert spread < 1.2, f"bead not compact (spread {spread:.2f})"
    print(f"  beads: OK  (nucleosome wrap is a compact coil, spread {spread:.2f})")

    # 4. the warp_nucleosome scene renders and animates (extended strand -> beads on a string)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    ext = np.asarray(ws.render("warp_nucleosome", width=160, height=90, time=0.0), np.float32)
    beads = np.asarray(ws.render("warp_nucleosome", width=160, height=90, time=3.4), np.float32)
    assert np.all(np.isfinite(ext)) and beads.max() > 0.1 and beads.std() > 0.01, "bad frame"
    assert np.abs(ext - beads).mean() > 1e-3, "warp_nucleosome: strand -> beads did not animate"
    print("  scene warp_nucleosome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
