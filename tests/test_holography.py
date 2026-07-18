"""Smoke tests for the AdS/CFT holography set (ads_cft disk + ads_bulk ray-traced bulk).

Both frames are deterministic, so beyond finite/in-range/animating we assert real
structure. ads_cft: a bright conformal-boundary ring at the disk rim, a tiled
(high-variance) hyperbolic bulk inside it, a dimmer — but not empty — holographic
exterior. ads_bulk: a genuine event-horizon shadow, a lit CFT boundary sky, and the
LOD tiers (boundary bounces + integration steps) rendering at every quality.

    python -m tests.test_holography
"""

import numpy as np
import warp as wp

import warp_shaders as ws
from warp_shaders import lod
from warp_shaders.engine.adscft import hawking_temperature

_DISK_R = 0.43  # keep in sync with warp_shaders/scenes/ads_cft.py


def _render(name, t, w=160, h=120):
    img = np.asarray(ws.render(name, width=w, height=h, time=t), np.float32)
    assert img.shape == (h, w, 3), (name, img.shape)
    assert np.all(np.isfinite(img)), f"{name}: non-finite"
    assert img.min() >= 0.0, f"{name}: negative"
    assert img.max() > 0.1, f"{name}: essentially black"
    assert img.std() > 0.02, f"{name}: flat fill"
    return img


def main():
    wp.init()

    a = _render("ads_cft", 2.0)
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
    b = _render("ads_cft", 6.0)
    assert np.abs(a - b).mean() > 1e-3, "isometry flow did not change the frame"
    print("  ads_cft flow: OK  (frames differ)")

    # ads_bulk — a real horizon shadow against a lit CFT boundary sky
    a = _render("ads_bulk", 3.0)
    lum = a.mean(axis=2)
    # bloom bleeds a little light into the shadow at smoke-test resolution, so the
    # "dark" threshold is 0.04 (empirically the shadow sits at ~0.10 frame fraction)
    dark = float((lum < 0.04).mean())
    assert dark > 0.05, f"ads_bulk: no horizon shadow (dark frac {dark:.3f})"
    assert float((lum > 0.15).mean()) > 0.25, "ads_bulk: boundary CFT sky not lit"
    b = _render("ads_bulk", 6.5)
    assert np.abs(a - b).mean() > 1e-3, "ads_bulk: orbit/flow did not change the frame"
    print(f"  ads_bulk t=3.0: OK  (shadow-frac {dark:.3f}  max {a.max():.3f})")

    # LOD contract: every quality tier renders (bounces 1..4, steps scale with tier)
    prev = lod.active_tier().name
    try:
        for q in ("low", "medium", "high", "ultra"):
            lod.set_active(q)
            _render("ads_bulk", 3.0, w=96, h=72)
        print("  ads_bulk LOD: OK  (low/medium/high/ultra all render)")
    finally:
        lod.set_active(prev)

    # Hawking temperature: T(r_h) = (L² + 3r_h²)/(4πL²r_h) has a MINIMUM at r_h = L/√3
    # (T_min = √3/2πL) — small AdS holes cool as they grow (negative specific heat),
    # large ones heat up (positive specific heat, the Hawking-Page structure).
    l_ads = 7.0
    t_small = hawking_temperature(0.5, l_ads)     # r_h ≈ 0.98, small branch
    t_mid = hawking_temperature(50.0, l_ads)      # r_h ≈ 16, near/above the minimum
    t_large = hawking_temperature(500.0, l_ads)   # r_h ≈ 37, large branch
    t_min = np.sqrt(3.0) / (2.0 * np.pi * l_ads)
    assert t_small > t_min and t_mid > t_min and t_large > t_min, "T below analytic minimum"
    assert t_small > t_mid, f"small branch not cooling with size ({t_small:.4f} vs {t_mid:.4f})"
    assert t_large > t_mid, f"large branch not heating with size ({t_large:.4f} vs {t_mid:.4f})"
    print(f"  hawking_temperature: OK  (T_min {t_min:.4f} < both branches; "
          f"small {t_small:.4f} > mid {t_mid:.4f} < large {t_large:.4f})")

    print("ALL PASSED (2 scenes + LOD sweep + thermodynamics)")


if __name__ == "__main__":
    main()
