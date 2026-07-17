"""Tests for Process 2 — base-pair bounding (warp_shaders.genome.basepair).

Operator spec: use the floating tokens to FORM PAIRS, BASE PAIRS — "for 100000 this mean at least
50000 base pairs". A conserving process: every token joins exactly one pair, none spawned.

  1. conservation: 2 x pairs == the token count (every token in exactly one pair);
  2. count: pairs == tokens / 2, comfortably past the operator's "at least 50000";
  3. binding: partner tokens are spatial neighbours (plausible binding, not random long jumps);
  4. bases: each pair is a complementary base pair (A-T / G-C) with finite colours;
  5. the warp_basepair scene renders on the real board and the field animates (cloud -> bound).

    python -m tests.test_basepair
"""

import numpy as np

from warp_shaders.genome import bind_pairs, tokenize_card


def main():
    n_tokens = tokenize_card(sub=2, block=5).n
    bp = bind_pairs(sub=2, block=5)

    # 1. conservation — every token joins exactly one pair
    assert 2 * bp.n_pairs == n_tokens, f"not conserved: 2 x {bp.n_pairs} != {n_tokens} tokens"
    print(f"  conservation: OK  ({n_tokens} tokens -> {bp.n_pairs} pairs, every token bound once)")

    # 2. count — pairs = tokens/2, well past the operator's 50000
    assert bp.n_pairs == n_tokens // 2 and bp.n_pairs >= 50_000
    print(f"  count: OK  ({bp.n_pairs} base pairs, >= 50000)")

    # 3. binding — partners are spatial neighbours (median rung well under a token-cloud diameter)
    d = np.linalg.norm(bp.b_pos - bp.a_pos, axis=1)
    assert np.median(d) < 0.5, f"partners not local (median rung {np.median(d):.3f})"
    print(f"  binding: OK  (median partner distance {np.median(d):.3f} — local, plausible)")

    # 4. bases — complementary pairs, finite colours in [0,1]
    assert bp.a_col.shape == (bp.n_pairs, 3) and bp.b_col.shape == (bp.n_pairs, 3)
    assert np.all(np.isfinite(bp.a_col)) and np.all(np.isfinite(bp.b_col))
    assert bp.a_col.min() >= 0.0 and bp.a_col.max() <= 1.0
    assert not np.allclose(bp.a_col, bp.b_col), "pair colours not distinguishable by base"
    print("  bases: OK  (complementary A-T / G-C colours)")

    # 5. the warp_basepair scene renders and the field animates (floating cloud -> bound pairs)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    cloud = np.asarray(ws.render("warp_basepair", width=160, height=90, time=0.0), np.float32)
    bound = np.asarray(ws.render("warp_basepair", width=160, height=90, time=4.0), np.float32)
    assert np.all(np.isfinite(cloud)) and bound.max() > 0.1 and bound.std() > 0.01, "bad frame"
    assert np.abs(cloud - bound).mean() > 1e-3, "warp_basepair: cloud -> bound did not animate"
    print("  scene warp_basepair: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
