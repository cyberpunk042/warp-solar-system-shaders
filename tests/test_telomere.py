"""Tests for Process 6 — telomeres (warp_shaders.genome.telomere), the honest, chained version.

Process 6 chains from Process 5's ACTUAL output (the 30 nm fibre) and curls the strand's two ends into
t-loop caps. A linear strand has exactly two ends, so exactly two telomeres — only the terminal base
pairs are reshaped; nothing spawned.

  1. chained + conserved: the input IS Process 5's fibre (fib_a/fib_b == coil_fibre), count unchanged;
  2. two real telomeres: exactly two ends tinted + reshaped, each terminal stretch curled into a compact
     t-loop (bounded lasso), the free tip tucked back near its anchor;
  3. the warp_telomere scene renders and the ends animate (fibre -> t-loops).

    python -m tests.test_telomere
"""

import numpy as np

from warp_shaders.genome import coil_fibre
from warp_shaders.genome.telomere import cap_telomeres


def main():
    fb = coil_fibre(sub=2, block=5)
    tl = cap_telomeres(sub=2, block=5)

    # 1. chained + conserved — the input is exactly Process 5's fibre
    assert tl.n_pairs == fb.n_pairs, f"pairs {tl.n_pairs} != {fb.n_pairs}"
    assert np.array_equal(tl.fib_a, fb.fib_a) and np.array_equal(tl.fib_b, fb.fib_b), \
        "Process 6 does not chain from Process 5's fibre"
    assert np.all(np.isfinite(tl.tel_a)) and np.all(np.isfinite(tl.tel_b))
    print(f"  chained + conserved: OK  ({tl.n_pairs} pairs, input == Process 5)")

    # 2. two real telomeres — exactly the two ends, curled into compact t-loops
    assert tl.ends.shape == (2, 3), "a linear strand has exactly two ends"
    assert int(tl.is_tel.sum()) == 2 * tl.tel_len, "telomere count != two terminal stretches"
    # only the ends are telomeric (the two terminal runs of the strand)
    idx = np.where(tl.is_tel)[0]
    assert idx.min() == 0 and idx.max() == tl.n_pairs - 1, "telomeres are not at the strand ends"
    # the terminal pairs were actually reshaped (curled), not left as the fibre
    moved = np.abs(tl.tel_a[tl.is_tel] - tl.fib_a[tl.is_tel]).mean()
    assert moved > 0.1, "telomere ends were not reshaped into loops"
    # each t-loop is a compact lasso (bounded), and the free tip tucks back near its anchor
    end0 = tl.tel_a[:tl.tel_len]
    span = float(np.ptp(end0, axis=0).max())
    assert span < 8.0, f"t-loop 0 not a compact lasso (span {span:.1f})"
    tip_to_anchor = float(np.linalg.norm(tl.tel_a[0] - tl.ends[0]))
    assert tip_to_anchor < 5.0, f"free tip did not tuck back (dist {tip_to_anchor:.1f})"
    print(f"  two t-loops: OK  ({tl.tel_len} bp each, lasso span {span:.1f}, tip tucks {tip_to_anchor:.1f})")

    # 3. the warp_telomere scene renders and the ends animate (fibre -> t-loops)
    import warp as wp
    import warp_shaders as ws
    wp.init()
    fibre = np.asarray(ws.render("warp_telomere", width=140, height=120, time=0.4), np.float32)
    looped = np.asarray(ws.render("warp_telomere", width=140, height=120, time=6.0), np.float32)
    assert np.all(np.isfinite(fibre)) and looped.max() > 0.1 and looped.std() > 0.01, "bad frame"
    assert np.abs(fibre - looped).mean() > 1e-3, "warp_telomere: ends -> t-loops did not animate"
    print("  scene warp_telomere: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
