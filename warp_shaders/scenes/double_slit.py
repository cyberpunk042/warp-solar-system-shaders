"""Double slit — the most famous experiment in physics, as a wave simulation.

A plane wave sweeps down onto a barrier pierced by two narrow slits. By **Huygens' principle**
each slit becomes a fresh source of circular waves, and on the far side those two wave-trains
**interfere** — reinforcing along some directions, cancelling along others — casting the
alternating bright/dark **fringes** that first proved light is a wave (and, with single quanta,
that everything is). Simulated for real (``sim/wave.py``): a line of in-phase oscillators drives
the plane wave, a reflecting wall with two gaps splits it, and the far-field fan of fringes is
lit like a ripple tank (caustic brightness tracks the surface curvature). See
``docs/research/41-waves-and-resonance.md``.
"""

import numpy as np

from ..engine import post
from ..scene import Scene
from ..sim.wave import WaveField

_N = 224
_WALL_Y = 0.4


def _render(width, height, time, mouse, device):
    n = _N
    if width * height <= 96 * 72:
        n = 96

    field = WaveField(n=n, c=0.5, damp=0.9986, border=0.17)
    field.add_line_source(0.12, amp=1.0, omega=0.5, x0=0.06, x1=0.94)
    field.double_slit(_WALL_Y, gap=0.028, sep=0.2, thickness=0.02)
    steps = 40 + int(time * 48.0)          # time drives the wave's arrival + fringe formation
    u = field.run(steps)
    lap = field.laplacian()

    caustic = np.clip(-lap * 24.0, 0.0, 3.0)
    water = np.array([0.02, 0.05, 0.09], np.float32)
    caust_col = np.array([0.5, 0.82, 1.0], np.float32)
    img = water[None, None, :] + caustic[..., None] * caust_col[None, None, :]

    # draw the barrier as a dark metal bar (where the wall mask is set)
    if field.wall is not None:
        img[field.wall] = np.array([0.06, 0.07, 0.09], np.float32)

    # upscale grid → frame (bilinear)
    gyv = np.linspace(0.0, n - 1.0, height)
    gxv = np.linspace(0.0, n - 1.0, width)
    y0 = np.floor(gyv).astype(np.int64); y1 = np.minimum(y0 + 1, n - 1); wy = (gyv - y0)[:, None, None]
    x0 = np.floor(gxv).astype(np.int64); x1 = np.minimum(x0 + 1, n - 1); wx = (gxv - x0)[None, :, None]
    top = img[y0][:, x0] * (1.0 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1.0 - wx) + img[y1][:, x1] * wx
    img = top * (1.0 - wy) + bot * wy

    return post.tonemap(img.astype(np.float32), mode="aces", exposure=1.3, preserve_hue=True)


SCENE = Scene(
    name="double_slit",
    description="the double-slit experiment as a real wave simulation — a plane wave driven onto "
                "a reflecting barrier with two slits, each slit re-radiating circular waves "
                "(Huygens) that interfere into the classic bright/dark fringe fan on the far "
                "side, lit like a ripple tank. Finite-difference wave equation with a barrier "
                "mask.",
    renderer=_render,
)
