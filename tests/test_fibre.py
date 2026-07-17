"""Tests for Process 5 — the 30 nm fibre (warp_shaders.genome.fibre), the honest, chained version.

Process 5 chains from Process 4's ACTUAL output (the nucleosome beads) and coils them into 30 nm solenoid
fibres — ~6 beads per turn, one row of beads per fibre, so the 1663 beads funnel into ~47 fibres. Each
bead moves as a rigid unit (its wrapped ring carried along); linker DNA is re-routed to the new
neighbours. Nothing spawned.

  1. chained + conserved: the input IS Process 4's beads (bead_a/bead_b == nuc_a/nuc_b), count unchanged;
  2. the funnel + solenoid are real: beads funnel into far fewer fibres, each a coil of the right radius,
     and the field compacts along the fibre axis;
  3. the warp_fibre scene renders and the whole field animates (beads -> fibres).

    python -m tests.test_fibre
"""

import numpy as np

from warp_shaders.genome import coil_fibre, wrap_nucleosomes


def main():
    nc = wrap_nucleosomes(sub=2, block=5)
    fb = coil_fibre(sub=2, block=5)

    # 1. chained + conserved — the input is exactly Process 4's beads-on-a-string
    assert fb.n_pairs == nc.n_pairs, f"pairs {fb.n_pairs} != {nc.n_pairs}"
    assert np.array_equal(fb.bead_a, nc.nuc_a) and np.array_equal(fb.bead_b, nc.nuc_b), \
        "Process 5 does not chain from Process 4's beads"
    assert np.all(np.isfinite(fb.fib_a)) and np.all(np.isfinite(fb.fib_b))
    print(f"  chained + conserved: OK  ({fb.n_pairs} pairs, input == Process 4)")

    # 2. the funnel + solenoid are real
    expected = (nc.n_beads + nc.grid_nx - 1) // nc.grid_nx
    assert fb.n_fibres == expected and fb.n_fibres < nc.n_beads, \
        f"no funnel: {nc.n_beads} beads -> {fb.n_fibres} fibres (expected {expected})"
    # the coil reaches the fibre radius (bead centres ring the axis)
    reach = float(np.max(np.abs(fb.centers[:, 1])))
    assert abs(reach - fb.fibre_radius) < 0.05, f"coil radius {reach:.2f} != {fb.fibre_radius}"
    # compaction along the fibre axis (x): the coiled band is far narrower than the bead carpet
    bead_w = float(np.ptp(fb.bead_a[:, 0]))
    fib_w = float(np.ptp(fb.fib_a[:, 0]))
    assert fib_w < 0.5 * bead_w, f"not compacted along the fibre ({fib_w:.1f} vs carpet {bead_w:.1f})"
    print(f"  funnel + solenoid: OK  ({nc.n_beads} beads -> {fb.n_fibres} fibres, "
          f"~{fb.beads_per_turn:.0f}/turn, band {fib_w:.0f} << carpet {bead_w:.0f})")

    # 3. the warp_fibre scene renders and the whole field animates (beads -> fibres)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    beads = np.asarray(ws.render("warp_fibre", width=120, height=150, time=0.4), np.float32)
    fibres = np.asarray(ws.render("warp_fibre", width=120, height=150, time=6.0), np.float32)
    assert np.all(np.isfinite(beads)) and fibres.max() > 0.1 and fibres.std() > 0.01, "bad frame"
    assert np.abs(beads - fibres).mean() > 1e-3, "warp_fibre: beads -> fibres did not animate"
    print("  scene warp_fibre: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
