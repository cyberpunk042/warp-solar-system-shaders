"""Tests for Process 4 — nucleosomes (warp_shaders.genome.nucleosome), the honest, chained version.

Process 4 chains from Process 3's ACTUAL output (the wound double helices) and wraps each into a
nucleosome bead — ~one helix's worth of DNA coiled ~1.75 turns around a histone core, its ends the linker
to the neighbouring beads. "Beads on a string" built from the same base pairs — nothing spawned.

  1. chained + conserved: the input IS Process 3's wound helices (helix_a/helix_b == wound_positions),
     one bead per helix, all finite, count unchanged;
  2. the wrap is real + compact: each bead coils around a core (ring of radius ~core_radius) and is far
     shorter vertically than the helix it came from (spatial compaction);
  3. the warp_nucleosome scene renders and the whole field animates (helices -> beads on a string).

    python -m tests.test_nucleosome
"""

import numpy as np

from warp_shaders.genome import wind_helix, wound_positions, wrap_nucleosomes


def main():
    hx = wind_helix(sub=2, block=5)
    pa, pb = wound_positions(hx)
    nc = wrap_nucleosomes(sub=2, block=5)

    # 1. chained + conserved — the input is exactly Process 3's wound-helix output
    assert nc.n_pairs == hx.n_pairs, f"pairs {nc.n_pairs} != {hx.n_pairs}"
    assert nc.n_beads == hx.n_helix, f"beads {nc.n_beads} != helices {hx.n_helix}"
    assert np.array_equal(nc.helix_a, pa) and np.array_equal(nc.helix_b, pb), \
        "Process 4 does not chain from Process 3's wound helices"
    assert np.all(np.isfinite(nc.nuc_a)) and np.all(np.isfinite(nc.nuc_b))
    print(f"  chained + conserved: OK  ({nc.n_pairs} pairs, {nc.n_beads} beads, input == Process 3)")

    # 2. the wrap is real + compact — a bead coils around its core and is much shorter than its helix
    g = nc.bp_per_nuc
    idx = np.arange(nc.n_pairs)
    for b in (0, nc.n_beads // 2, nc.n_beads - 2):     # sample a few beads
        m = (idx // g) == b
        c = nc.centers[b]
        rad = np.hypot(nc.nuc_a[m, 0] - c[0], nc.nuc_a[m, 2] - c[2])
        wrapped = rad[(nc.link_frac < np.linspace(0, 1, m.sum())) &
                      (np.linspace(0, 1, m.sum()) < 1 - nc.link_frac)]
        assert wrapped.size and abs(np.median(wrapped) - nc.core_radius) < 0.12, \
            f"bead {b} DNA does not wrap the core (median r {np.median(wrapped):.2f})"
    bead0 = (idx // g) == 0
    helix_h = float(np.ptp(nc.helix_a[bead0, 1]))
    bead_h = float(np.ptp(nc.nuc_a[bead0, 1]))
    assert bead_h < 0.5 * helix_h, f"bead not compacted vertically ({bead_h:.2f} vs helix {helix_h:.2f})"
    print(f"  wrap real + compact: OK  (DNA rings the ~{nc.core_radius} core; bead {bead_h:.2f} << "
          f"helix {helix_h:.2f} tall)")

    # 3. the warp_nucleosome scene renders and the whole field animates (helices -> beads)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    helices = np.asarray(ws.render("warp_nucleosome", width=120, height=150, time=0.4), np.float32)
    beads = np.asarray(ws.render("warp_nucleosome", width=120, height=150, time=6.0), np.float32)
    assert np.all(np.isfinite(helices)) and beads.max() > 0.1 and beads.std() > 0.01, "bad frame"
    assert np.abs(helices - beads).mean() > 1e-3, "warp_nucleosome: helices -> beads did not animate"
    print("  scene warp_nucleosome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
