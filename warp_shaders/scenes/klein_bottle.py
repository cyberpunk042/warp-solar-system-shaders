"""A Klein bottle — a surface with no inside or outside.

The **Klein bottle** is a closed surface that passes through its own wall — it has
no distinct inside or outside (only embeddable without self-intersection in 4D).
Its classic 3-D shadow is the **figure-8 immersion**: a tube swept around a circle
while its cross-section rotates a half-turn (like a Möbius band closing on itself).
Sampled and splatted, coloured around the sweep. See
``docs/research/27-mathematics-made-visible.md``. --frames orbits it.
"""

import math

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene


def _klein(nu=300, nv=76, R=2.3, scale=0.46):
    pts = np.empty((nu * nv, 3), np.float32)
    cols = np.empty((nu * nv, 3), np.float32)
    idx = 0
    for iu in range(nu):
        u = 2.0 * math.pi * iu / nu
        cu = math.cos(u / 2.0)
        su = math.sin(u / 2.0)
        f = iu / nu
        base = np.array([0.35 + 0.55 * math.cos(f * 6.283),
                         0.5 + 0.4 * math.cos(f * 6.283 + 2.1),
                         0.65 + 0.35 * math.cos(f * 6.283 + 4.2)])
        for iv in range(nv):
            v = 2.0 * math.pi * iv / nv
            sv = math.sin(v)
            s2v = math.sin(2.0 * v)
            r = R + cu * sv - su * s2v
            x = r * math.cos(u)
            y = r * math.sin(u)
            z = su * sv + cu * s2v
            pts[idx] = np.array([x, y, z]) * scale
            shade = 0.55 + 0.45 * math.cos(v)          # gives the tube some form
            cols[idx] = np.clip(base * shade, 0.04, 1.0)
            idx += 1
    return pts, cols


_PTS, _COLS = _klein()


def _render(width, height, time, mouse, device):
    hdr = splat_scene(_PTS, _COLS, width, height, time, device, foc=2.0, dist=3.7,
                      el=0.28, az_speed=0.16, intensity=0.04)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="klein_bottle",
    description="A Klein bottle (figure-8 immersion) — a closed surface with no inside "
                "or outside, drawn as a tube swept around a circle while its section "
                "rotates a half-turn, coloured around the sweep. --frames orbits it.",
    renderer=_render,
)
