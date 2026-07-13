"""A bar magnet — the dipole field made visible.

A bar magnet is a **magnetic dipole**: field lines loop out of the **north** pole,
around, and into the **south**, closing through the magnet — the pattern iron filings
trace. There are no magnetic monopoles; the lines always close on themselves. See
``docs/research/32-electromagnetism-and-fields.md``.
"""

import math

import numpy as np

from ..engine import post
from ..fields.draw2d import integrate_line, draw_polyline
from ..scene import Scene

_N = (0.0, 0.55)       # north pole
_S = (0.0, -0.55)      # south pole
_AX = 1.15


def _field(p):
    rn = (p[0] - _N[0], p[1] - _N[1])
    rs = (p[0] - _S[0], p[1] - _S[1])
    dn = (rn[0] ** 2 + rn[1] ** 2) ** 1.5 + 1e-6
    ds = (rs[0] ** 2 + rs[1] ** 2) ** 1.5 + 1e-6
    return (rn[0] / dn - rs[0] / ds, rn[1] / dn - rs[1] / ds)


def _lines():
    lines = []
    for k in range(16):
        a = 2.0 * math.pi * (k + 0.5) / 16.0
        start = (_N[0] + 0.14 * math.cos(a), _N[1] + 0.14 * math.sin(a))
        lines.append(integrate_line(_field, start, 0.02, 600, bounds=1.5,
                                    stop_r=0.13, stops=[_S]))
    return lines


_LINES = _lines()


def _render(width, height, time, mouse, device):
    W, H = int(width), int(height)
    hdr = np.zeros((H, W, 3), np.float32)
    hdr[:, :] = np.array([0.02, 0.02, 0.028], np.float32)
    for ln in _LINES:
        draw_polyline(hdr, ln, (0.55, 0.72, 1.0), max(1.2, W * 0.0032), _AX, glow=0.9)

    # the magnet bar: N (red) top half, S (blue) bottom half
    ys, xs = np.mgrid[0:H, 0:W]
    wx = (xs / W * 2.0 - 1.0) * _AX
    wy = 1.0 - ys / H * 2.0
    barx = np.abs(wx) < 0.16
    bary = np.abs(wy) < 0.62
    bar = barx & bary
    north = bar & (wy >= 0.0)
    south = bar & (wy < 0.0)
    hdr[north] = np.array([0.85, 0.16, 0.14], np.float32)
    hdr[south] = np.array([0.2, 0.35, 0.9], np.float32)
    # pole faces brighter
    edge = bar & (np.abs(np.abs(wx) - 0.16) < 0.012)
    hdr[edge] += np.array([0.3, 0.3, 0.35], np.float32)

    r = max(2, int(min(W, H) * 0.006))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="bar_magnet",
    description="A bar magnet's dipole field — glowing field lines looping out of the "
                "north pole (red), around, and back into the south (blue), closing "
                "through the magnet, as iron filings trace them.",
    renderer=_render,
)
