"""Bioluminescence — a cloud of living light in the midnight zone.

Most deep-sea life makes its own light: luciferase oxidises luciferin to release a
cold blue-green photon (~470–490 nm, the colour that travels furthest in seawater).
Here a drifting cloud of plankton and small creatures sparks and pulses in otherwise
total darkness, swept by slow currents. See ``docs/research/28-the-deep-ocean.md``.
--frames drifts and flashes the swarm.
"""

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene

_RNG = np.random.default_rng(7)
_N = 3200
# base positions in a squashed sphere + per-point pulse phase/speed
_u = _RNG.normal(size=(_N, 3)).astype(np.float32)
_u /= (np.linalg.norm(_u, axis=1, keepdims=True) + 1e-6)
_rad = (_RNG.random(_N) ** 0.5).astype(np.float32) * 2.3
_BASE = _u * _rad[:, None] * np.array([1.3, 1.0, 1.3], np.float32)
_PH = (_RNG.random(_N) * 6.2831).astype(np.float32)
_SP = (0.6 + _RNG.random(_N) * 2.4).astype(np.float32)
_HUE = _RNG.random(_N).astype(np.float32)          # 0=green .. 1=cyan/blue
_FREQ = (0.3 + _RNG.random((_N, 3)) * 0.5).astype(np.float32)


def _render(width, height, time, mouse, device):
    # slow current drift
    drift = 0.18 * np.sin(_BASE * _FREQ * 2.0 + time * 0.5)
    pts = _BASE + drift
    # per-point bioluminescent pulse (some flash brightly)
    pulse = 0.25 + 0.75 * np.clip(np.sin(time * _SP + _PH), 0.0, 1.0) ** 3
    green = np.array([0.15, 0.95, 0.7], np.float32)
    blue = np.array([0.2, 0.6, 1.0], np.float32)
    cols = (green[None, :] * (1.0 - _HUE[:, None]) + blue[None, :] * _HUE[:, None])
    bright = 1.0 + 3.0 * (_HUE < 0.12)             # a few are big bright creatures
    cols = cols * (pulse * bright)[:, None] * 2.4
    hdr = splat_scene(pts.astype(np.float32), cols.astype(np.float32), width, height,
                      time, device, foc=1.9, dist=4.2, el=0.12, az_speed=0.06,
                      intensity=0.16, bg=(0.004, 0.008, 0.016))
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=0.7, strength=0.7, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="bioluminescent",
    description="A cloud of living light — deep-sea plankton and creatures sparking "
                "cold blue-green (luciferin/luciferase) in total darkness, drifting on "
                "slow currents and flashing. --frames drifts and flashes the swarm.",
    renderer=_render,
)
