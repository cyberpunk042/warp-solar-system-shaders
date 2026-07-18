"""Process 6 — the telomere t-loop cap, as a SOLID lit strand (not points).

Chains from Process 5's actual output: it takes the real capped strand ``cap_telomeres`` produced — the 30 nm
fibre whose terminal stretch curls back into a **t-loop** lasso that protects the end — and renders the true
conserved backbone as a tube **mesh**, ray-traced solid by the engine. We feature one real end up close: the
telomere repeat (tinted telomere-green) leaving the coiled fibre and looping back on itself. Nothing invented,
the real lib positions given a surface.
"""

from __future__ import annotations

import numpy as np

from ..genome import cap_telomeres
from ..genome.render import render_strand
from ..genome.tube import tube_mesh
from ..scene import Scene

_TL = cap_telomeres(sub=2, block=5)
_L = int(_TL.tel_len)


def _smooth(x, w):
    k = np.ones(w) / w
    pad = np.pad(x, ((w, w), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, d], k, mode="same")[w:-w] for d in range(3)], 1)


# feature end 0: the t-loop (pairs [0, L)) plus a stretch of the coiled fibre leading into it. Take the real
# conserved backbone centreline straight from Process 6's output, smooth lightly (keep the loop + coil, drop
# the sub-nucleosome noise), centre on the loop, colour the telomere repeat green.
_N = _L + 4200
_seg = (0.5 * (_TL.tel_a + _TL.tel_b))[:_N]
_seg = _seg - _seg[:_L].mean(axis=0)                              # centre on the t-loop
_S = 3
_center = _smooth(_seg, 150)[::_S].astype(np.float64)            # clean tube (nucleosome bumps removed)

_istel = np.zeros(_N, bool)
_istel[:_L] = True
_it = _istel[::_S][: _center.shape[0]]
_green = np.array([0.42, 0.95, 0.55], np.float32)
_violet = np.array([0.70, 0.58, 0.90], np.float32)
_COL = np.where(_it[:, None], _green[None, :], _violet[None, :]).astype(np.float32)
_MESH = tube_mesh(_center, radius=0.34, color=_COL, sides=16)


def _camera(time: float):
    ang = 0.7 + 0.52 * float(time)                                # slow orbit (a turn over the gif)
    r = 11.5
    tgt = np.array([0.0, 0.0, 0.0], np.float32)
    eye = tgt + np.array([r * np.sin(ang), 3.2, r * np.cos(ang)], np.float32)
    return eye, tgt


def _render(width, height, time, mouse, device):
    eye, tgt = _camera(float(time))
    img = render_strand(_MESH, int(width), int(height), eye, tgt,
                        sun_dir=(0.45, 0.72, 0.55), device=device, fov=40.0, exposure=1.12)
    return np.clip(img, 0.0, 1.0)


SCENE = Scene(
    name="warp_telomere",
    description=(
        "Process 6 — the telomere t-loop, as a solid lit strand. The real conserved backbone of the capped "
        "strand end (from cap_telomeres) is tubed and ray-traced by the engine on a dark specimen background: "
        "the telomere repeat (green) leaves the coiled 30 nm fibre and loops back into the protective t-loop "
        "lasso — the actual lib positions given a surface, not points."
    ),
    renderer=_render,
)
