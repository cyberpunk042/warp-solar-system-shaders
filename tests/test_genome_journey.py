"""Tests for the genome journey (warp_shaders.genome.journey + scene warp_genome).

The journey chains all six conserving processes into one morph: the same 182872 base pairs carried
tokens -> base pairs -> double helix -> nucleosomes -> 30nm fibre -> chromosome.

  1. conservation: six keyframes, one point per base pair in each, all finite;
  2. common frame: every stage is normalised to a comparable size (so one camera frames them all);
  3. distinct stages: consecutive keyframes are genuinely different shapes;
  4. the warp_genome scene renders and the morph animates across the stages.

    python -m tests.test_genome_journey
"""

import numpy as np

from warp_shaders.genome import bind_pairs, genome_journey, STAGE_NAMES


def main():
    n_pairs = bind_pairs(sub=2, block=5).n_pairs
    kf, colors, blue = genome_journey(sub=2, block=5)

    # 1. conservation — six stages, one point per base pair, finite
    assert kf.shape == (6, n_pairs, 3), f"unexpected keyframes shape {kf.shape}"
    assert len(STAGE_NAMES) == 6
    assert np.all(np.isfinite(kf)) and colors.shape == (n_pairs, 3) and np.all(np.isfinite(colors))
    print(f"  conservation: OK  (6 stages x {n_pairs} points, none spawned)")

    # 2. common frame — every stage normalised to a comparable size
    extents = [np.linalg.norm(kf[k].max(0) - kf[k].min(0)) for k in range(6)]
    assert max(extents) < 4.0 * min(extents), f"stages not comparably sized: {np.round(extents, 1)}"
    print(f"  common frame: OK  (stage extents {np.round(extents, 1)})")

    # 3. distinct stages — consecutive keyframes differ
    for k in range(5):
        d = np.abs(kf[k] - kf[k + 1]).mean()
        assert d > 0.1, f"stage {STAGE_NAMES[k]} -> {STAGE_NAMES[k+1]} not distinct (mean {d:.3f})"
    print("  distinct stages: OK  (each morph moves the points)")

    # 4. the warp_genome scene renders and animates across the stages
    import warp as wp
    import warp_shaders as ws
    wp.init()
    early = np.asarray(ws.render("warp_genome", width=120, height=120, time=0.5), np.float32)
    late = np.asarray(ws.render("warp_genome", width=120, height=120, time=6.2), np.float32)
    assert np.all(np.isfinite(early)) and late.max() > 0.1 and late.std() > 0.01, "bad frame"
    assert np.abs(early - late).mean() > 1e-3, "warp_genome: journey did not animate"
    print("  scene warp_genome: OK")

    print("ALL PASSED")


if __name__ == "__main__":
    main()
