"""Smoke tests for the relativity masterpieces (kerr, binary_bh, wormhole_dive).

Each is a deterministic geodesic render. We assert finite, in-range output that is
non-degenerate and animates (two different time steps differ), plus a scene-specific
structural check: the black-hole scenes must contain a genuine dark shadow, and the
wormhole must contain a bright Einstein-ring / portal region.

    python -m tests.test_relativity
"""

import numpy as np
import warp as wp

import warp_shaders as ws


def _render(name, t):
    img = np.asarray(ws.render(name, width=120, height=90, time=t), np.float32)
    assert img.shape == (90, 120, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.1, f"{name}: essentially black"
    assert img.std() > 0.02, f"{name}: flat fill"
    return img


def main():
    wp.init()

    # kerr — spinning hole: a real dark shadow occupies a meaningful fraction of the frame
    a = _render("kerr", 0.5)
    dark = float((a.mean(axis=2) < 0.02).mean())
    assert dark > 0.05, f"kerr: no shadow (dark frac {dark:.3f})"
    b = _render("kerr", 1.5)
    assert np.abs(a - b).mean() > 1e-3, "kerr: camera orbit did not change the frame"
    print(f"  kerr: OK  (max {a.max():.3f} shadow-frac {dark:.3f})")

    # binary_bh — two shadows: even more dark area, and it animates through the inspiral
    a = _render("binary_bh", 0.8)
    dark = float((a.mean(axis=2) < 0.02).mean())
    assert dark > 0.08, f"binary_bh: no shadows (dark frac {dark:.3f})"
    b = _render("binary_bh", 3.0)
    assert np.abs(a - b).mean() > 1e-3, "binary_bh: inspiral did not change the frame"
    print(f"  binary_bh: OK  (max {a.max():.3f} shadow-frac {dark:.3f})")

    # wormhole_dive — bright portal ring present; the dive changes the view across the throat
    a = _render("wormhole_dive", 0.0)
    bright = float((a.mean(axis=2) > 0.4).mean())
    assert bright > 0.02, f"wormhole_dive: no bright portal ring (frac {bright:.3f})"
    b = _render("wormhole_dive", 7.5)
    assert np.abs(a - b).mean() > 1e-3, "wormhole_dive: dive did not change the frame"
    print(f"  wormhole_dive: OK  (max {a.max():.3f} ring-frac {bright:.3f})")

    print("ALL PASSED (3 scenes, 2 frames each)")


if __name__ == "__main__":
    main()
