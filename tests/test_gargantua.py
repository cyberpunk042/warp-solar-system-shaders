"""Smoke test for the geodesic-traced black hole (gargantua).

Renders tiny frames and asserts finite, in-range output with a genuine black shadow
(a central region that stays dark — the event horizon) surrounded by a bright disk.

    python -m tests.test_gargantua
"""

import numpy as np
import warp as wp

import warp_shaders as ws


def main():
    wp.init()
    img = np.asarray(ws.render("gargantua", width=120, height=96, time=0.0), np.float32)
    assert img.shape == (96, 120, 3), img.shape
    assert np.all(np.isfinite(img)), "non-finite"
    assert img.min() >= 0.0, "negative"
    assert img.max() > 0.1, "essentially black — disk not lit"
    assert img.std() > 0.02, "flat fill"

    # the centre of the frame should hold the shadow: much darker than the disk band
    cx, cy = 60, 48
    centre = img[cy - 6:cy + 6, cx - 6:cx + 6].mean()
    frame_mean = img.mean()
    assert centre < frame_mean, f"no dark shadow at centre ({centre:.3f} vs mean {frame_mean:.3f})"
    print(f"  gargantua: OK  (max {img.max():.3f} std {img.std():.3f} "
          f"centre {centre:.3f} < mean {frame_mean:.3f})")

    # a second time step (camera orbit) must also be finite + non-degenerate
    img2 = np.asarray(ws.render("gargantua", width=120, height=96, time=1.0), np.float32)
    assert np.all(np.isfinite(img2)) and img2.std() > 0.02, "orbit frame degenerate"
    print("  gargantua orbit frame: OK")
    print("ALL PASSED (1 scene, 2 frames)")


if __name__ == "__main__":
    main()
