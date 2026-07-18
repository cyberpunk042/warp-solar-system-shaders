"""Smoke tests for the AdS/CFT holography scene (ads_cft).

The frame is deterministic, so beyond finite/in-range/animating we can assert real
structure: a bright conformal-boundary ring at the disk rim, a tiled (high-variance)
hyperbolic bulk inside it, and a dimmer — but not empty — holographic exterior.

    python -m tests.test_holography
"""

import numpy as np
import warp as wp

import warp_shaders as ws

_DISK_R = 0.43  # keep in sync with warp_shaders/scenes/ads_cft.py


def _render(t, w=160, h=120):
    img = np.asarray(ws.render("ads_cft", width=w, height=h, time=t), np.float32)
    assert img.shape == (h, w, 3), img.shape
    assert np.all(np.isfinite(img)), "ads_cft: non-finite"
    assert img.min() >= 0.0, "ads_cft: negative"
    assert img.max() > 0.1, "ads_cft: essentially black"
    assert img.std() > 0.02, "ads_cft: flat fill"
    return img


def main():
    wp.init()

    a = _render(2.0)
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # same mapping as the kernel: uv in units of height, disk boundary at r = 1
    u = (xx + 0.5 - 0.5 * w) / h
    v = (0.5 * h - (yy + 0.5)) / h
    r = np.sqrt(u * u + v * v) / _DISK_R
    lum = a.mean(axis=2)

    ring = lum[(r > 0.96) & (r < 1.04)].mean()
    bulk = lum[r < 0.85].mean()
    ext = lum[(r > 1.15) & (r < 1.8)].mean()
    # ACES tonemapping compresses highlights, so assert clear margins rather than raw ratios
    assert ring > 1.2 * bulk, f"no conformal-boundary ring (ring {ring:.3f} vs bulk {bulk:.3f})"
    assert ring > 1.2 * ext, f"ring not brighter than exterior ({ring:.3f} vs {ext:.3f})"
    assert lum[r < 0.85].std() > 0.05, "bulk shows no tiling structure"
    assert ext > 0.01, "holographic exterior is empty"
    assert bulk > ext * 0.8, f"bulk unexpectedly dimmer than far exterior ({bulk:.3f} vs {ext:.3f})"
    print(f"  ads_cft t=2.0: OK  (ring {ring:.3f}  bulk {bulk:.3f}  ext {ext:.3f})")

    # the Mobius isometry flow must animate the frame
    b = _render(6.0)
    assert np.abs(a - b).mean() > 1e-3, "isometry flow did not change the frame"
    print("  ads_cft flow: OK  (frames differ)")

    print("ALL PASSED (1 scene, 2 frames)")


if __name__ == "__main__":
    main()
