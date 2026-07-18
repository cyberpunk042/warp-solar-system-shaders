"""Tests for the continuous-compression master animation (warp_shaders.scenes.warp_genome).

The scene is ONE thread of the real base pairs whose shape is a nested supercoil driven by a single
condensation parameter c. It is a genuine continuous coil — not a lerp between poses — so the invariants are
about the physics of the coil, not keyframes:

  1. the thread is the real conserved base pairs (count), finite everywhere, at every c;
  2. it is a real COMPRESSION — the thread's extent SHRINKS monotonically as c rises (the axis shortens as it
     coils), and it never grows;
  3. the warp_genome scene renders and animates (extended thread -> packed chromatid).

    python -m tests.test_genome
"""

import numpy as np


def main():
    from warp_shaders.scenes import warp_genome as G

    # 1. the conserved thread — real base-pair count, two strands, finite at every c
    assert G._N == 182872, f"unexpected base-pair count {G._N}"
    heights = []
    for c in np.linspace(0.0, 1.0, 9):
        a, b, half = G._positions(float(c))
        assert a.shape == b.shape == (G._N, 3), f"bad strand shape {a.shape} at c={c}"
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), f"non-finite positions at c={c}"
        heights.append(half)
    print(f"  conserved thread: OK  ({G._N} base pairs x 2 strands, finite at every c)")

    # 2. real compression — the thread only ever gets SHORTER as it condenses (never grows)
    h = np.array(heights)
    assert np.all(np.diff(h) <= 1e-4), f"thread grew during condensation (not compression): {h}"
    assert h[0] > 3.0 * h[-1], f"not enough compression: extended {h[0]:.1f} vs condensed {h[-1]:.1f}"
    print(f"  compression: OK  (thread shrinks {h[0]:.0f} -> {h[-1]:.1f}, monotonic)")

    # 3. the scene renders and animates the whole coil
    import warp as wp
    import warp_shaders as ws
    wp.init()
    start = np.asarray(ws.render("warp_genome", width=140, height=160, time=0.8), np.float32)   # extended
    end = np.asarray(ws.render("warp_genome", width=140, height=160, time=10.5), np.float32)    # chromatid
    assert np.all(np.isfinite(start)) and end.max() > 0.1 and end.std() > 0.01, "bad frame"
    assert np.abs(start - end).mean() > 1e-3, "warp_genome: the coil did not animate"
    print("  scene warp_genome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
