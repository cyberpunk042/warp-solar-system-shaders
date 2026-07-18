"""Process 6 — the telomere: the fibre's end CURLS into a protective t-loop cap.

Chains from Process 5's actual output. ``cap_telomeres`` supplies both real end states: the straight 30 nm
fibre end (``fib_a``) and the same strand with its terminal stretch curled into a **t-loop** lasso (``tel_a``).
This scene animates the real **transition** between them — every terminal base pair moves continuously from
its fibre position to its t-loop position (the same reshaping the process makes; the loop hangs to the side so
the strand never passes through itself) — and renders it SOLID on a dark specimen background. Not a spin: the
end visibly curls into the cap.
"""

from __future__ import annotations

import numpy as np

from ..genome import cap_telomeres
from ..genome.render import render_strand
from ..genome.tube import tube_mesh
from ..scene import Scene

_TL = cap_telomeres(sub=2, block=5)
_L = int(_TL.tel_len)
_N = _L + 4200


def _smooth(x, w):
    k = np.ones(w) / w
    pad = np.pad(x, ((w, w), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, d], k, mode="same")[w:-w] for d in range(3)], 1)


# feature end 0: the straight fibre end (fib) and the curled t-loop (tel), both from Process 6's real output,
# same pairs / same frame. Subtract one shared offset (the loop centre), smooth to clean tubes.
_fib = _TL.fib_a[:_N].astype(np.float64)
_tel = _TL.tel_a[:_N].astype(np.float64)
_off = _tel[:_L].mean(axis=0)
_S = 3
_FIB = _smooth(_fib - _off, 150)[::_S]                           # straight fibre end
_LOOP = _smooth(_tel - _off, 90)[::_S]                           # the curled t-loop
_BEND = _smooth(_tel - _off, 700)[::_S]                          # the loop smoothed to a gentle bend (its "axis")
_CURL = _LOOP - _BEND                                            # the sharp loop detail (develops as it curls)

_istel = np.zeros(_N, bool)
_istel[:_L] = True
_it = _istel[::_S][: _FIB.shape[0]]
_green = np.array([0.42, 0.95, 0.55], np.float32)
_violet = np.array([0.70, 0.58, 0.90], np.float32)
_COL = np.where(_it[:, None], _green[None, :], _violet[None, :]).astype(np.float32)

_DUR = 4.0


def _ss(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _render(width, height, time, mouse, device):
    t = float(time)
    # the end first bends over (fibre -> gentle bend), THEN the loop tightens and tucks — a natural curling,
    # not a straight-line morph; every frame a valid partially-curled end.
    wa = _ss((t - 0.3) / (_DUR * 0.55))
    wc = _ss((t - 0.3 - _DUR * 0.35) / (_DUR * 0.6))
    center = (1.0 - wa) * _FIB + wa * _BEND + wc * _CURL
    mesh = tube_mesh(center, radius=0.34, color=_COL, sides=14)
    eye = np.array([6.5, 3.0, 10.5], np.float32)                 # fixed 3/4 view; the curling is the motion
    tgt = np.array([0.0, -0.6, 0.0], np.float32)
    img = render_strand(mesh, int(width), int(height), eye, tgt,
                        sun_dir=(0.45, 0.72, 0.55), device=device, fov=42.0, exposure=1.12)
    return np.clip(img, 0.0, 1.0)


SCENE = Scene(
    name="warp_telomere",
    description=(
        "Process 6 — the telomere t-loop. The straight 30 nm fibre end curls into a protective t-loop lasso: "
        "this scene animates the real transition between cap_telomeres' two end states (fib_a -> tel_a), the "
        "terminal telomere repeat (green) moving continuously from the fibre into the side-hanging loop (so "
        "the strand never threads itself), tubed and ray-traced solid — the end visibly curling into the cap."
    ),
    renderer=_render,
)
