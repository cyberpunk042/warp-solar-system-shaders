"""Tests for Process 3 — the double helix (warp_shaders.genome.helix).

Operator spec: the base pairs wind into DNA — each pair a rung, its two tokens the two backbones. A
conserving process: every base pair placed exactly once, nothing spawned.

  1. conservation: rungs == base pairs == tokens/2 (every pair placed once);
  2. geometry: the two backbones sit on a cylinder of the helix radius (constant distance from axis);
  3. twist: consecutive rungs advance by a constant angle (~10.5 bp/turn) and rise (a real helix);
  4. the warp_helix scene renders on the real board and the strand animates (loose ladder -> wound).

    python -m tests.test_helix
"""

import numpy as np

from warp_shaders.genome import bind_pairs, wind_helix


def main():
    n_pairs = bind_pairs(sub=2, block=5).n_pairs
    hx = wind_helix(sub=2, block=5, radius=0.72, rise=0.095)

    # 1. conservation — one rung per base pair
    assert hx.n_pairs == n_pairs, f"rungs {hx.n_pairs} != base pairs {n_pairs}"
    print(f"  conservation: OK  ({hx.n_pairs} rungs for {n_pairs} base pairs, none spawned)")

    # 2. geometry — both backbones lie on the helix cylinder (radius 0.72 in x/z about the axis)
    r1 = np.hypot(hx.s1[:, 0], hx.s1[:, 2])
    r2 = np.hypot(hx.s2[:, 0], hx.s2[:, 2])
    assert np.allclose(r1, 0.72, atol=1e-3) and np.allclose(r2, 0.72, atol=1e-3)
    print("  geometry: OK  (both backbones on the helix cylinder)")

    # 3. twist — constant rise + constant angular step (a true right-handed helix)
    dy = np.diff(hx.s1[:1000, 1])
    assert np.allclose(dy, dy[0], atol=1e-5) and dy[0] > 0, "rise not constant / not climbing"
    ang = np.arctan2(hx.s1[:1000, 2], hx.s1[:1000, 0])
    dstep = np.diff(np.unwrap(ang))
    assert np.allclose(dstep, dstep[0], atol=1e-4), "twist not constant"
    turns_per_bp = abs(dstep[0]) / (2 * np.pi)
    print(f"  twist: OK  (constant rise + ~{1/turns_per_bp:.1f} bp/turn)")

    # 4. the warp_helix scene renders and the strand animates (loose ladder -> wound helix)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    loose = np.asarray(ws.render("warp_helix", width=160, height=90, time=0.0), np.float32)
    wound = np.asarray(ws.render("warp_helix", width=160, height=90, time=3.4), np.float32)
    assert np.all(np.isfinite(loose)) and wound.max() > 0.1 and wound.std() > 0.01, "bad frame"
    assert np.abs(loose - wound).mean() > 1e-3, "warp_helix: ladder -> helix did not animate"
    print("  scene warp_helix: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
