"""Smoke test for the warp_chromosome animation scenes.

Verifies the compression-in-time visualization renders finite frames, that the raw strand really
has more beads than the coiled strand (the animation compresses), and that different times give
different frames (it actually animates).

    python -m tests.test_warp_chromosome
"""

import numpy as np
import warp as wp

import warp_shaders as ws
from warp_shaders.scenes import warp_chromosome as wch


def main():
    wp.init()

    # the layout must shrink the strand as compaction progresses (bead count raw > coiled)
    raw_pos, _, _, raw_path = wch._layout("dna", 0.0)
    coil_pos, _, _, coil_path = wch._layout("dna", 1.0)
    assert len(raw_pos) > len(coil_pos), \
        f"coiling did not reduce bead count ({len(raw_pos)} -> {len(coil_pos)})"
    assert len(raw_path) == len(coil_path) > 0, "backbone polyline missing"
    print(f"  layout compresses: OK  (beads {len(raw_pos)} raw -> {len(coil_pos)} coiled)")

    # renders are finite, in range, and non-degenerate at both ends of the animation
    for name in ("warp_chromosome", "warp_fold_text", "warp_fold_bytes"):
        a = np.asarray(ws.render(name, width=120, height=90, time=0.3), np.float32)
        b = np.asarray(ws.render(name, width=120, height=90, time=4.0), np.float32)
        for img in (a, b):
            assert img.shape == (90, 120, 3)
            assert np.all(np.isfinite(img)) and img.min() >= 0.0, f"{name}: bad pixels"
            assert img.max() > 0.1 and img.std() > 0.01, f"{name}: degenerate"
        assert np.abs(a - b).mean() > 1e-3, f"{name}: frame did not change over time"
        print(f"  {name}: OK  (animates raw->coiled)")

    print("ALL PASSED (3 scenes)")


if __name__ == "__main__":
    main()
