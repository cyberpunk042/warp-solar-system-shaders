"""Tests for Process 3 — the double helix (warp_shaders.genome.helix), the honest, chained version.

Process 3 chains from Process 2's ACTUAL output: it winds the base-pair field (each pair's two tokens
on a rung) into the DNA double helix — no regeneration, the two tokens become the two backbones.

  1. chained + conserved: the helix input IS Process 2's field (field_a/field_b from bind_pairs), one
     rung per base pair, all finite;
  2. helix parameters: a positive per-base-pair twist over the whole strand;
  3. the warp_helix scene renders and the whole strand animates (base-pair field -> wound helix).

    python -m tests.test_helix
"""

import numpy as np

from warp_shaders.genome import bind_pairs, wind_helix


def main():
    bp = bind_pairs(sub=2, block=5)
    hx = wind_helix(sub=2, block=5)

    # 1. chained + conserved — the helix's input is exactly Process 2's ordered field
    assert hx.n_pairs == bp.n_pairs, f"pairs {hx.n_pairs} != base pairs {bp.n_pairs}"
    assert np.array_equal(hx.field_a, bp.field_a) and np.array_equal(hx.field_b, bp.field_b), \
        "Process 3 does not chain from Process 2's actual field"
    assert np.all(np.isfinite(hx.field_a)) and np.all(np.isfinite(hx.field_b))
    # the field is a set of vertical rungs: field_b sits above field_a by 2*HL
    dy = (hx.field_b - hx.field_a)[:, 1]
    assert np.allclose(dy, dy[0], atol=1e-4) and dy[0] > 0, "input is not the base-pair rung field"
    print(f"  chained + conserved: OK  ({hx.n_pairs} rungs, input == Process 2 field)")

    # 2. helix parameters — a real, positive twist per base pair over the whole strand
    assert hx.dtheta > 0.0 and hx.radius > 0.0 and hx.height > 0.0
    total_turns = hx.dtheta * hx.n_pairs / (2 * np.pi)
    assert total_turns > 4.0, f"too few turns for a helix ({total_turns:.1f})"
    print(f"  helix parameters: OK  (~{total_turns:.0f} turns, radius {hx.radius})")

    # 3. the warp_helix scene renders and the whole strand animates (field -> ladder -> wound helix)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    field = np.asarray(ws.render("warp_helix", width=120, height=150, time=0.4), np.float32)
    wound = np.asarray(ws.render("warp_helix", width=120, height=150, time=6.0), np.float32)
    assert np.all(np.isfinite(field)) and wound.max() > 0.1 and wound.std() > 0.01, "bad frame"
    assert np.abs(field - wound).mean() > 1e-3, "warp_helix: field -> helix did not animate"
    print("  scene warp_helix: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
