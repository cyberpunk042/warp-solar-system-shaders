"""An electric dipole — field lines and equipotentials.

Two opposite charges make a **dipole**: electric field lines run *from* the positive
charge *to* the negative, while **equipotential** surfaces (of constant voltage) nest
around each charge, everywhere perpendicular to the field. Unlike magnetic poles, the
sources here — the charges — can be pulled apart. See
``docs/research/32-electromagnetism-and-fields.md``.
"""

import math

import numpy as np

from ..engine import post
from ..fields.draw2d import integrate_line, draw_polyline, draw_point
from ..scene import Scene

_P = (-0.6, 0.0)       # positive charge
_N = (0.6, 0.0)        # negative charge
_AX = 1.3


def _field(p):
    rp = (p[0] - _P[0], p[1] - _P[1])
    rn = (p[0] - _N[0], p[1] - _N[1])
    dp = (rp[0] ** 2 + rp[1] ** 2) ** 1.5 + 1e-6
    dn = (rn[0] ** 2 + rn[1] ** 2) ** 1.5 + 1e-6
    return (rp[0] / dp - rn[0] / dn, rp[1] / dp - rn[1] / dn)


def _lines():
    lines = []
    for k in range(18):
        a = 2.0 * math.pi * (k + 0.5) / 18.0
        start = (_P[0] + 0.12 * math.cos(a), _P[1] + 0.12 * math.sin(a))
        lines.append(integrate_line(_field, start, 0.02, 700, bounds=1.7,
                                    stop_r=0.11, stops=[_N]))
    return lines


_LINES = _lines()


def _render(width, height, time, mouse, device):
    W, H = int(width), int(height)
    hdr = np.zeros((H, W, 3), np.float32)
    hdr[:, :] = np.array([0.02, 0.02, 0.03], np.float32)

    # faint equipotential rings (contours of V = 1/rP - 1/rN)
    ys, xs = np.mgrid[0:H, 0:W]
    wx = (xs / W * 2.0 - 1.0) * _AX
    wy = 1.0 - ys / H * 2.0
    rp = np.hypot(wx - _P[0], wy - _P[1]) + 0.04
    rn = np.hypot(wx - _N[0], wy - _N[1]) + 0.04
    V = 1.0 / rp - 1.0 / rn
    bands = 0.5 + 0.5 * np.cos(np.clip(V, -9.0, 9.0) * 3.2)
    hdr += (np.array([0.14, 0.15, 0.2], np.float32)[None, None, :]
            * (bands ** 6.0)[:, :, None])

    for ln in _LINES:
        draw_polyline(hdr, ln, (1.0, 0.92, 0.55), max(1.2, W * 0.003), _AX, glow=0.8)

    # the two charges
    draw_point(hdr, _P, (1.4, 0.5, 0.3), max(4.0, W * 0.014), _AX)
    draw_point(hdr, _N, (0.4, 0.6, 1.5), max(4.0, W * 0.014), _AX)
    # + / - glyphs
    def stamp(cx, cy, minus_only):
        px = int((cx / _AX * 0.5 + 0.5) * W)
        py = int((0.5 - cy * 0.5) * H)
        s = max(3, int(W * 0.012))
        hdr[py - 1:py + 2, px - s:px + s] = np.array([1.0, 1.0, 1.0], np.float32)
        if not minus_only:
            hdr[py - s:py + s, px - 1:px + 2] = np.array([1.0, 1.0, 1.0], np.float32)
    stamp(_P[0], _P[1], False)
    stamp(_N[0], _N[1], True)

    r = max(2, int(min(W, H) * 0.006))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="electric_dipole",
    description="An electric dipole — glowing field lines running from the positive "
                "charge (+) to the negative (−), with faint nested equipotential rings "
                "perpendicular to them.",
    renderer=_render,
)
