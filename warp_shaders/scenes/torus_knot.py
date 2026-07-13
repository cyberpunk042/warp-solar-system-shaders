"""A torus knot — a curve knotted on the surface of a torus.

A **(p,q) torus knot** winds *p* times around the torus's main axis and *q* times
through its hole; (2,3) is the **trefoil**, the simplest non-trivial knot. Here the
knot is tubed with a ring of points and splatted as a glowing closed loop, coloured
along its length. See ``docs/research/27-mathematics-made-visible.md``. --frames
orbits it.
"""

import math

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene


def _knot(p=2, q=3, samples=1700, ring=16, R=2.0, a=0.85, tube=0.26, scale=0.5):
    ts = np.linspace(0.0, 2.0 * math.pi, samples, endpoint=False)
    pts = np.empty((samples * ring, 3), np.float32)
    cols = np.empty((samples * ring, 3), np.float32)
    idx = 0
    for t in ts:
        ph = p * t
        ps = q * t
        rr = R + a * math.cos(ps)
        c = np.array([rr * math.cos(ph), rr * math.sin(ph), a * math.sin(ps)])
        # tangent (numeric) → normal frame
        dt = 1e-3
        ph2 = p * (t + dt); ps2 = q * (t + dt)
        rr2 = R + a * math.cos(ps2)
        c2 = np.array([rr2 * math.cos(ph2), rr2 * math.sin(ph2), a * math.sin(ps2)])
        T = c2 - c
        T /= (np.linalg.norm(T) + 1e-9)
        up = np.array([0.0, 1.0, 0.0]) if abs(T[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        N = np.cross(T, up); N /= (np.linalg.norm(N) + 1e-9)
        B = np.cross(T, N)
        f = t / (2.0 * math.pi)
        col = np.array([0.4 + 0.6 * math.cos(f * 6.283),
                        0.5 + 0.4 * math.cos(f * 6.283 + 2.1),
                        0.6 + 0.4 * math.cos(f * 6.283 + 4.2)])
        col = np.clip(col, 0.05, 1.0)
        for k in range(ring):
            u = 2.0 * math.pi * k / ring
            pt = c + tube * (math.cos(u) * N + math.sin(u) * B)
            pts[idx] = pt * scale
            cols[idx] = col
            idx += 1
    return pts, cols


_PTS, _COLS = _knot()


def _render(width, height, time, mouse, device):
    hdr = splat_scene(_PTS, _COLS, width, height, time, device, foc=2.0, dist=3.6,
                      el=0.35, az_speed=0.18, intensity=0.05)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.55, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.08)


SCENE = Scene(
    name="torus_knot",
    description="A (2,3) torus knot — the trefoil — a curve knotted on a torus, winding "
                "twice around the axis and three times through the hole, tubed and "
                "splatted as a glowing closed loop coloured along its length. "
                "--frames orbits it.",
    renderer=_render,
)
