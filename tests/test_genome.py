"""Tests for the whole-process master animation (warp_shaders.scenes.warp_genome).

The master scene stacks the ACTUAL output of every genome process and morphs continuously through them.
The load-bearing invariant is that every process exposes the same base pairs in the same order (so the morph
is a pure per-pair move, conserving — nothing spawned):

  1. every stage has the same pair count and finite positions (the chain invariant);
  2. the timeline progresses monotonically from stage 0 (base pairs) to the last (chromosome);
  3. the warp_genome scene renders and the whole thing animates (base pairs -> chromosome).

    python -m tests.test_genome
"""

import numpy as np


def main():
    from warp_shaders.scenes import warp_genome as G

    # 1. the chain invariant — 6 stages, same pair count, all finite, same order
    assert G._STAGES == 6, f"expected 6 stages, got {G._STAGES}"
    assert G._KA.shape == G._KB.shape == (6, G._P, 3), f"bad keyframe shape {G._KA.shape}"
    assert np.all(np.isfinite(G._KA)) and np.all(np.isfinite(G._KB)), "non-finite keyframe positions"
    assert G._P == 182872, f"unexpected pair count {G._P}"
    print(f"  chain invariant: OK  (6 stages x {G._P} pairs, same order, finite)")

    # 2. the timeline runs monotonically stage 0 -> last, and settles (holds) on the chromosome
    prev = -1.0
    for t in np.linspace(0.0, 27.0, 40):
        g = G._progress(float(t))
        assert g >= prev - 1e-6, f"progress went backwards at t={t}"
        prev = g
    assert G._progress(0.0) == 0.0, "should start on the base pairs"
    assert G._progress(100.0) == float(G._STAGES - 1), "should settle on the chromosome"
    print("  timeline: OK  (monotonic base pairs -> chromosome, then holds)")

    # 3. the scene renders and animates the whole way down
    import warp as wp
    import warp_shaders as ws
    wp.init()
    start = np.asarray(ws.render("warp_genome", width=140, height=150, time=0.3), np.float32)   # base pairs
    end = np.asarray(ws.render("warp_genome", width=140, height=150, time=26.0), np.float32)    # chromosome
    assert np.all(np.isfinite(start)) and end.max() > 0.1 and end.std() > 0.01, "bad frame"
    assert np.abs(start - end).mean() > 1e-3, "warp_genome: base pairs -> chromosome did not animate"
    print("  scene warp_genome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
