"""Tests for Process 5 — the 30nm fibre (warp_shaders.genome.fiber).

Operator spec: the beads-on-a-string coil into a solenoid — the 30nm chromatin fibre. A conserving
process: each nucleosome bead carried whole onto the fibre, nothing spawned.

  1. conservation: one point per base pair (== tokens/2), string and fibre both present;
  2. packing: the fibre is shorter end-to-end than the string (the string coiled up);
  3. solenoid: the fibre points wind around the axis at ~constant radius (a real coil);
  4. the warp_fiber scene renders and animates (beads-on-a-string -> coiled fibre).

    python -m tests.test_fiber
"""

import numpy as np

from warp_shaders.genome import bind_pairs, coil_fiber


def main():
    n_pairs = bind_pairs(sub=2, block=5).n_pairs
    fb = coil_fiber(sub=2, block=5)

    # 1. conservation
    assert fb.n_pairs == n_pairs, f"points {fb.n_pairs} != base pairs {n_pairs}"
    assert fb.string.shape == (n_pairs, 3) and fb.fiber.shape == (n_pairs, 3)
    assert np.all(np.isfinite(fb.string)) and np.all(np.isfinite(fb.fiber))
    print(f"  conservation: OK  ({fb.n_pairs} points, {fb.n_beads} beads carried onto the fibre)")

    # 2. packing — coiling shortens the strand end-to-end
    str_len = fb.string[:, 0].max() - fb.string[:, 0].min()
    fib_len = fb.fiber[:, 0].max() - fb.fiber[:, 0].min()
    assert fib_len < str_len, f"coiling did not shorten ({fib_len:.0f} vs {str_len:.0f})"
    print(f"  packing: OK  (string {str_len:.0f} -> fibre {fib_len:.0f} along the axis)")

    # 3. solenoid — the fibre winds around the x-axis at a bounded radius (not a straight line)
    r = np.hypot(fb.fiber[:, 1], fb.fiber[:, 2])
    assert np.percentile(r, 50) > 1.0, "fibre is not wound around an axis"
    assert np.percentile(r, 99) < 6.0, "fibre radius unbounded"
    print(f"  solenoid: OK  (wound at radius ~{np.percentile(r, 50):.1f}, a real coil)")

    # 4. the warp_fiber scene renders and animates (string -> coiled fibre)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    s = np.asarray(ws.render("warp_fiber", width=160, height=90, time=0.0), np.float32)
    c = np.asarray(ws.render("warp_fiber", width=160, height=90, time=3.4), np.float32)
    assert np.all(np.isfinite(s)) and c.max() > 0.1 and c.std() > 0.01, "bad frame"
    assert np.abs(s - c).mean() > 1e-3, "warp_fiber: string -> fibre did not animate"
    print("  scene warp_fiber: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
