"""Fluid — a rising smoke column from a 2-D Navier–Stokes solver.

A real incompressible fluid simulation (``sim/fluid.py``, Jos Stam's *Stable Fluids*): a
velocity field made divergence-free every step by a Poisson pressure projection, moved by
semi-Lagrangian advection, with buoyancy lifting the hot fluid and vorticity confinement
restoring the curls that keep the smoke billowing. A hot emitter at the base drives a
turbulent, rising, curling column. The dye field is mapped to warm smoke over a glowing
ember base. Over ``--frames`` the plume develops and rolls; a still catches it mid-rise. See
``docs/research/40-physics-sims.md``.
"""

import numpy as np

from ..engine import post
from ..scene import Scene
from ..sim.fluid import StableFluid

_N = 220
_DT = 0.09


def _ramp(x, c0, c1):
    x = x[..., None]
    return c0[None, None, :] * (1.0 - x) + c1[None, None, :] * x


def _render(width, height, time, mouse, device):
    n = _N
    if width * height <= 96 * 72:
        n = 64

    sim = StableFluid(n=n, buoy=2.5, vort=9.0, seed=3)
    steps = 108 + int(time * 22.0)                     # still (t=0) sits mid-rise
    d, t, vx, vy = sim.run(steps, _DT)

    d = np.clip(d, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)

    # smoke: cool grey-blue body (kept dim so it reads as smoke, not light); embers glow at base
    smoke_lo = np.array([0.05, 0.06, 0.09], np.float32)
    smoke_hi = np.array([0.4, 0.42, 0.5], np.float32)
    img = _ramp(d, smoke_lo, smoke_hi) * d[..., None]
    fire = np.array([3.4, 1.5, 0.35], np.float32)
    glow = (np.clip(t - 0.4, 0.0, 1.0) ** 1.7)[..., None] * fire[None, None, :]
    img = img + glow

    # bilinear upscale from the sim grid to the requested frame
    gy = np.linspace(0.0, n - 1.0, height)
    gx = np.linspace(0.0, n - 1.0, width)
    y0 = np.floor(gy).astype(np.int64); y1 = np.minimum(y0 + 1, n - 1); wy = (gy - y0)[:, None, None]
    x0 = np.floor(gx).astype(np.int64); x1 = np.minimum(x0 + 1, n - 1); wx = (gx - x0)[None, :, None]
    top = img[y0][:, x0] * (1.0 - wx) + img[y0][:, x1] * wx
    bot = img[y1][:, x0] * (1.0 - wx) + img[y1][:, x1] * wx
    img = top * (1.0 - wy) + bot * wy

    return post.tonemap(img.astype(np.float32), mode="aces", exposure=1.5, preserve_hue=True)


SCENE = Scene(
    name="fluid",
    description="a rising smoke column from a real 2-D incompressible Navier–Stokes solver "
                "(Stam's Stable Fluids) — divergence-free pressure projection, semi-Lagrangian "
                "advection, buoyancy and vorticity confinement drive a turbulent curling plume "
                "off a glowing ember base. Grid-based fluid dynamics.",
    renderer=_render,
)
