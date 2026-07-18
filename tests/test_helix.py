"""Tests for Process 3 — the double helices (warp_shaders.genome.helix), the honest, chained version.

Process 3 chains from Process 2's ACTUAL output: it winds the base-pair field (each pair's two tokens
on a rung) into DNA double helices — no regeneration, the two tokens become the two backbones. A double
helix only holds ~100 base pairs, so the pairs are grouped (``bp_per_helix`` each) into MANY short
helices, not one giant one.

  1. chained + conserved: the input IS Process 2's field (field_a/field_b from bind_pairs), one rung per
     base pair, all finite;
  2. many helices with real proportions: n_helix ≈ n_pairs / bp_per_helix, ~10.5 base pairs per turn;
  3. the warp_helix scene renders and the whole field animates (base-pair field -> wound helices).

    python -m tests.test_helix
"""

import numpy as np

from warp_shaders.genome import bind_pairs, wind_helix


def main():
    bp = bind_pairs(sub=2, block=5)
    hx = wind_helix(sub=2, block=5)

    # 1. chained + conserved — the input is exactly Process 2's ordered field
    assert hx.n_pairs == bp.n_pairs, f"pairs {hx.n_pairs} != base pairs {bp.n_pairs}"
    assert np.array_equal(hx.field_a, bp.field_a) and np.array_equal(hx.field_b, bp.field_b), \
        "Process 3 does not chain from Process 2's actual field"
    assert np.all(np.isfinite(hx.field_a)) and np.all(np.isfinite(hx.field_b))
    # the field is a set of vertical rungs: field_b sits above field_a by 2*HL
    dy = (hx.field_b - hx.field_a)[:, 1]
    assert np.allclose(dy, dy[0], atol=1e-4) and dy[0] > 0, "input is not the base-pair rung field"
    print(f"  chained + conserved: OK  ({hx.n_pairs} rungs, input == Process 2 field)")

    # 2. MANY helices with real B-DNA proportions (not one giant helix)
    assert hx.n_helix > 1, "should be many double helices, not one"
    expected = (hx.n_pairs + hx.bp_per_helix - 1) // hx.bp_per_helix
    assert hx.n_helix == expected, f"n_helix {hx.n_helix} != ceil(pairs/bp_per_helix) {expected}"
    assert hx.centers.shape == (hx.n_helix, 3) and np.all(np.isfinite(hx.centers))
    assert hx.dtheta > 0.0 and hx.radius > 0.0 and hx.height > 0.0
    turns_per_helix = hx.bp_per_helix * hx.dtheta / (2 * np.pi)
    assert 4.0 < turns_per_helix < 20.0, f"unphysical turns per helix ({turns_per_helix:.1f})"
    # ~10.5 base pairs per turn is the real B-DNA value
    assert abs(hx.bp_per_helix / turns_per_helix - 10.5) < 0.5, "not ~10.5 base pairs per turn"
    print(f"  many helices: OK  ({hx.n_helix} helices, {hx.bp_per_helix} bp each, "
          f"~{turns_per_helix:.1f} turns/helix)")

    # 3. the warp_helix scene renders and the whole field animates (field -> ladders -> wound helices)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    field = np.asarray(ws.render("warp_helix", width=120, height=150, time=0.4), np.float32)
    wound = np.asarray(ws.render("warp_helix", width=120, height=150, time=6.0), np.float32)
    assert np.all(np.isfinite(field)) and wound.max() > 0.1 and wound.std() > 0.01, "bad frame"
    assert np.abs(field - wound).mean() > 1e-3, "warp_helix: field -> helices did not animate"
    print("  scene warp_helix: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
