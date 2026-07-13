"""A tesseract — the shadow of a rotating 4-cube.

A **tesseract** (4-cube) has 16 vertices in four-dimensional space. We can't see 4D,
but we can **rotate** the cube in a 4D plane and **project** it down to 3D (then to
the screen) — just as a wireframe cube on paper is a 3-cube's shadow. As it turns,
the inner cube appears to swell *through* the outer one and turn inside-out: a shadow
of a motion we cannot directly see. See ``docs/research/27-mathematics-made-visible.md``.
--frames turns it in 4D.
"""

import itertools
import math

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene

_V4 = np.array(list(itertools.product([-1.0, 1.0], repeat=4)), np.float32)
_EDGES = [(i, k) for i in range(16) for k in range(i + 1, 16)
          if np.sum(np.abs(_V4[i] - _V4[k])) == 2.0]     # differ in one coordinate


def _rot(a, b, ang):
    m = np.eye(4, dtype=np.float32)
    c, s = math.cos(ang), math.sin(ang)
    m[a, a] = c; m[a, b] = -s; m[b, a] = s; m[b, b] = c
    return m


def _project(time):
    m = _rot(0, 3, time * 0.5) @ _rot(1, 2, time * 0.34)   # rotate in 4D
    v = _V4 @ m.T
    k = 1.0 / (2.7 - v[:, 3])                               # 4D -> 3D perspective
    v3 = v[:, :3] * k[:, None]
    return v3, k


def _build(time, per=74):
    v3, k = _project(time)
    kn = (k - k.min()) / (k.max() - k.min() + 1e-6)
    pts = np.empty((len(_EDGES) * per, 3), np.float32)
    cols = np.empty((len(_EDGES) * per, 3), np.float32)
    idx = 0
    cool = np.array([0.2, 0.5, 1.0])
    warm = np.array([1.0, 0.55, 0.2])
    for (a, b) in _EDGES:
        ta = np.linspace(0.0, 1.0, per)
        for t in ta:
            p = v3[a] * (1.0 - t) + v3[b] * t
            kk = kn[a] * (1.0 - t) + kn[b] * t
            pts[idx] = p * 0.85
            cols[idx] = cool * (1.0 - kk) + warm * kk
            idx += 1
    return pts, cols


def _render(width, height, time, mouse, device):
    pts, cols = _build(time)
    hdr = splat_scene(pts, cols, width, height, time, device, foc=2.05, dist=3.6,
                      el=0.3, az_speed=0.08, intensity=0.085,
                      bg=(0.012, 0.016, 0.03))
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.6, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="tesseract",
    description="A tesseract (4-cube) rotating in a 4D plane and projected to 3D — the "
                "inner cube swells through the outer one and turns inside-out, a shadow "
                "of a motion in four dimensions we cannot directly see. Edges coloured "
                "by 4D depth. --frames turns it.",
    renderer=_render,
)
