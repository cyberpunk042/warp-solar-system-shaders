"""Ripple tank — two-source interference from a real wave-equation simulation.

The classic physics-classroom demo, simulated for real (``sim/wave.py``): two oscillators dip
into a shallow water tank in phase, each radiating circular waves, and where the waves overlap
they **interfere** — crest-on-crest reinforces (antinodes) and crest-on-trough cancels (nodes),
carving the fixed **hyperbolic nodal lines** that fan out between the sources. Lit from above
like a real ripple tank: the curved water surface focuses light into bright **caustic** fringes
(the brightness tracks the surface curvature, i.e. the Laplacian of the height), so the
interference pattern glows in cyan on dark water. Over ``--frames`` the waves radiate and the
pattern settles. See ``docs/research/41-waves-and-resonance.md``.
"""

import numpy as np

from ..engine import post
from ..scene import Scene
from ..sim.wave import WaveField

_N = 260


def _render(width, height, time, mouse, device):
    n = _N
    if width * height <= 96 * 72:
        n = 96

    field = WaveField(n=n, c=0.5, damp=0.9986, border=0.12)
    sep = 0.16
    field.add_source(0.5 - sep, 0.5, amp=1.0, omega=0.36)
    field.add_source(0.5 + sep, 0.5, amp=1.0, omega=0.36)
    steps = 175 + int(time * 9.0)
    u = field.run(steps)
    lap = field.laplacian()

    # caustic focus: bright where the surface is concave (light converges)
    caustic = np.clip(-lap * 26.0, 0.0, 3.2)
    # surface glint from the height gradient toward an upper-left light
    gx = 0.5 * (np.roll(u, -1, 1) - np.roll(u, 1, 1))
    gy = 0.5 * (np.roll(u, -1, 0) - np.roll(u, 1, 0))
    glint = np.clip(-(gx * 0.6 + gy * 0.6) * 6.0, 0.0, 1.0) ** 2

    water = np.array([0.015, 0.05, 0.08], np.float32)
    caust_col = np.array([0.45, 0.85, 1.0], np.float32)
    glint_col = np.array([0.7, 0.9, 1.0], np.float32)
    img = (water[None, None, :]
           + caustic[..., None] * caust_col[None, None, :]
           + glint[..., None] * glint_col[None, None, :] * 0.5)

    # the two emitters as bright dots
    yy, xx = np.mgrid[0:n, 0:n]
    for sx in (0.5 - sep, 0.5 + sep):
        d2 = (xx / n - sx) ** 2 + (yy / n - 0.5) ** 2
        img += np.exp(-d2 / (2.0 * (0.006 ** 2)))[..., None] * np.array([1.2, 1.3, 1.4], np.float32)

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
    name="ripple_tank",
    description="two-source wave interference from a real 2-D wave-equation simulation — two "
                "in-phase oscillators radiating circular waves that interfere into fixed "
                "hyperbolic nodal lines, lit like a ripple tank so the surface curvature focuses "
                "cyan caustic fringes on dark water. Finite-difference leapfrog with absorbing "
                "borders.",
    renderer=_render,
)
