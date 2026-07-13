"""Chladni figures — sand leaping to the nodal lines of a vibrating plate (cymatics).

Bow a metal plate, or drive it at a resonant frequency, and the sand scattered on top dances
away from the thrashing **antinodes** and piles up along the **nodal lines** — the curves that
stay still. Those curves are the zero set of a **standing-wave eigenmode** of the plate: for a
square plate the classic superposition

    f(x, y) = cos(nπx)·cos(mπy) − cos(mπx)·cos(nπy)

whose ``f = 0`` contour is exactly where the sand collects. Each integer pair ``(n, m)`` is a
resonance; raise the driving frequency and the plate jumps to a finer mode, the old pattern
dissolving into a new one. This scene renders that directly — a dark steel plate, warm grains
banked along the nodal set — and over ``--frames`` sweeps the frequency up so the figure keeps
re-forming. See ``docs/research/41-waves-and-resonance.md``.
"""

import numpy as np

from ..engine import post
from ..scene import Scene

# a curated ladder of resonances the frequency sweep climbs through
_MODES = [(2, 1), (3, 2), (4, 3), (5, 2), (5, 4), (6, 3), (7, 4), (6, 5), (8, 5), (9, 4)]


def _field(x, y, n, m):
    pi = np.pi
    return (np.cos(n * pi * x) * np.cos(m * pi * y)
            - np.cos(m * pi * x) * np.cos(n * pi * y))


def _render(width, height, time, mouse, device):
    # square plate in [0,1]²; look straight down so the pattern reads cleanly
    ax = np.linspace(0.0, 1.0, width, dtype=np.float32)
    ay = np.linspace(0.0, 1.0, height, dtype=np.float32)
    x, y = np.meshgrid(ax, ay)

    # sweep along the mode ladder; blend between neighbours for a smooth morph.
    # the still (time=0) lands exactly on a rich resonance rather than mid-morph.
    pos = (6.0 + float(mouse[0]) * 0.01 + time * 1.1) % float(len(_MODES))
    i0 = int(np.floor(pos)) % len(_MODES)
    i1 = (i0 + 1) % len(_MODES)
    fr = pos - np.floor(pos)
    n0, m0 = _MODES[i0]
    n1, m1 = _MODES[i1]
    f = (1.0 - fr) * _field(x, y, n0, m0) + fr * _field(x, y, n1, m1)

    # sand banks where the plate is still (f≈0); thin, crisp nodal lines
    sigma = 0.085
    sand = np.exp(-(f * f) / (2.0 * sigma * sigma))
    sand = np.clip(sand, 0.0, 1.0) ** 1.2

    # granular texture so it reads as grains, not a painted line
    rng = np.random.default_rng(12345)
    grain = 0.72 + 0.28 * rng.random((height, width)).astype(np.float32)
    sand = sand * grain

    # dark brushed-steel plate with a faint radial sheen + rim vignette
    r = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    sheen = 0.035 + 0.02 * np.cos(r * 26.0) * np.exp(-r * 2.2)
    vignette = np.clip(1.0 - (r * 1.35) ** 3, 0.0, 1.0)
    plate = np.stack([sheen, sheen * 1.02, sheen * 1.1], axis=-1)

    sand_col = np.array([1.15, 0.92, 0.55], np.float32)   # warm brass grains
    img = plate + sand[..., None] * sand_col[None, None, :]
    img = img * vignette[..., None]

    return post.tonemap(img.astype(np.float32), mode="aces", exposure=1.35, preserve_hue=True)


SCENE = Scene(
    name="chladni",
    description="Chladni figures (cymatics) — sand banking along the nodal lines of a vibrating "
                "square plate, the zero set of a standing-wave eigenmode cos(nπx)cos(mπy) − "
                "cos(mπx)cos(nπy). Warm brass grains on dark steel; over frames the driving "
                "frequency sweeps up so the resonance pattern keeps dissolving and re-forming.",
    renderer=_render,
)
