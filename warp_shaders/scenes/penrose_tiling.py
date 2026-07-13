"""A Penrose tiling — aperiodic order with five-fold symmetry.

A **Penrose tiling** covers the plane with two rhombi (a *thin* and a *fat*) so the
pattern **never repeats**, yet has perfect five-fold symmetry — *aperiodic order*.
Built here by **deflation**: start from a wheel of Robinson triangles around the
centre and repeatedly subdivide each with golden-ratio cuts. See
``docs/research/27-mathematics-made-visible.md``. --frames slowly rotates it.
"""

import cmath
import math

import numpy as np

from ..engine import post
from ..scene import Scene

_PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _wheel():
    tris = []
    for i in range(10):
        b = cmath.rect(1.0, (2 * i - 1) * math.pi / 10.0)
        c = cmath.rect(1.0, (2 * i + 1) * math.pi / 10.0)
        if i % 2 == 0:
            b, c = c, b
        tris.append((0, 0 + 0j, b, c))
    return tris


def _subdivide(tris):
    out = []
    for color, a, b, c in tris:
        if color == 0:                            # fat half-tile
            p = a + (b - a) / _PHI
            out.append((0, c, p, b))
            out.append((1, p, c, a))
        else:                                     # thin half-tile
            q = b + (a - b) / _PHI
            r = b + (c - b) / _PHI
            out.append((1, r, c, a))
            out.append((1, q, r, b))
            out.append((0, r, q, a))
    return out


def _tiling(depth=5):
    t = _wheel()
    for _ in range(depth):
        t = _subdivide(t)
    return t


_TILES = _tiling()


def _render(width, height, time, mouse, device):
    W, H = int(width), int(height)
    hdr = np.zeros((H, W, 3), np.float32)
    edge = np.zeros((H, W), np.float32)
    cx, cy = W * 0.5, H * 0.5
    scale = min(W, H) * 0.54
    rot = cmath.rect(1.0, time * 0.05)
    fat = np.array([0.42, 0.30, 0.12], np.float32)     # warm gold rhombus
    thin = np.array([0.10, 0.20, 0.34], np.float32)    # cool blue rhombus

    for color, a, b, c in _TILES:
        A = a * rot; B = b * rot; C = c * rot
        px = np.array([cx + A.real * scale, cx + B.real * scale, cx + C.real * scale])
        py = np.array([cy - A.imag * scale, cy - B.imag * scale, cy - C.imag * scale])
        minx = max(int(px.min()) - 1, 0); maxx = min(int(px.max()) + 2, W)
        miny = max(int(py.min()) - 1, 0); maxy = min(int(py.max()) + 2, H)
        if maxx <= minx or maxy <= miny:
            continue
        gx, gy = np.meshgrid(np.arange(minx, maxx), np.arange(miny, maxy))
        x0, y0, x1, y1, x2, y2 = px[0], py[0], px[1], py[1], px[2], py[2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-9:
            continue
        aa = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / den
        bb = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / den
        cc = 1.0 - aa - bb
        inside = (aa >= 0.0) & (bb >= 0.0) & (cc >= 0.0)
        colr = fat if color == 0 else thin
        sub = hdr[miny:maxy, minx:maxx]
        sub[inside] = colr * (0.6 + 0.4 * (color == 0))
        # glowing rhombus sides: the two edges from apex A (skip the B-C diagonal)
        for (ex0, ey0, ex1, ey1) in ((x0, y0, x1, y1), (x0, y0, x2, y2)):
            dx, dy = ex1 - ex0, ey1 - ey0
            ll = dx * dx + dy * dy + 1e-9
            tt = np.clip(((gx - ex0) * dx + (gy - ey0) * dy) / ll, 0.0, 1.0)
            dxp = gx - (ex0 + tt * dx); dyp = gy - (ey0 + tt * dy)
            d2 = dxp * dxp + dyp * dyp
            edge[miny:maxy, minx:maxx] = np.maximum(
                edge[miny:maxy, minx:maxx], np.exp(-d2 * 0.5))

    hdr += edge[:, :, None] * np.array([0.5, 0.85, 1.0], np.float32) * 1.6
    r = max(2, int(min(W, H) * 0.006))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="penrose_tiling",
    description="A Penrose tiling — two rhombi (thin blue, fat gold) covering the plane "
                "with perfect five-fold symmetry but never repeating (aperiodic order), "
                "built by golden-ratio deflation of Robinson triangles, edges glowing. "
                "--frames slowly rotates it.",
    renderer=_render,
)
