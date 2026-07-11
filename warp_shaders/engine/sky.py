"""Sky backgrounds — a reusable procedural starfield and a faint galaxy band.

The ecosystem's original `earthgfx.stars` is a single flat-blue twinkle layer;
this is the engine's richer, reusable version:

- `starfield(rd)` — **two** star layers (a sparse bright set + a dense faint one,
  so star *sizes* vary) whose **colour temperature** varies across the sky (warm
  ambers through hot blue-whites, via `color.kelvin_to_rgb`) instead of one flat
  tint.
- `milky_way(rd, axis, intensity)` — a soft fBm band of unresolved stars along a
  great circle, for a galactic-plane glow behind a scene.

Both are `@wp.func`s to call from your own kernel's background/miss path.
"""

from __future__ import annotations

import warp as wp

from ..procedural.noise import fbm3, value3
from .color import kelvin_to_rgb


@wp.func
def starfield(rd: wp.vec3) -> wp.vec3:
    """Procedural stars for a background ray `rd` (unit). Two size layers +
    spatially varying colour temperature."""
    # sparse bright stars + dense faint stars (different frequencies -> sizes)
    b1 = wp.pow(value3(rd * 180.0), 40.0) * 4.2
    b2 = wp.pow(value3(rd * 430.0 + wp.vec3(7.0, 3.0, 11.0)), 62.0) * 2.0
    s = b1 + b2
    # colour temperature drifts across the sky: ~3200 K amber .. ~16000 K blue
    warm = value3(rd * 60.0 + wp.vec3(19.0, 5.0, 2.0))
    col = kelvin_to_rgb(3200.0 + warm * 12800.0)
    return col * s


@wp.func
def milky_way(rd: wp.vec3, axis: wp.vec3, intensity: float) -> wp.vec3:
    """A soft galactic band: unresolved-star glow concentrated near the plane
    perpendicular to `axis`, dust-mottled with fBm."""
    lat = wp.abs(wp.dot(wp.normalize(rd), wp.normalize(axis)))
    band = wp.exp(-(lat / 0.22) * (lat / 0.22))          # bright near the plane
    dust = fbm3(rd * 8.0, 5)
    glow = band * (0.4 + 0.9 * dust) * intensity
    tint = wp.vec3(0.75, 0.78, 0.95) * (0.7 + 0.5 * dust)
    return tint * glow
