"""N-body gravity — two star clouds colliding under their own gravity.

A real O(N²) gravitational simulation (``sim/nbody.py``): every particle pulls on every other
by Newton's law (softened at short range), advanced with a leapfrog integrator, the whole force
matrix evaluated in parallel on Warp each step. Two Plummer clumps — one cool-blue, one warm-
gold — fall together, throw off curving **tidal tails**, and settle into a single relaxed cluster
with a blazing dense core. The particles are additively splatted so density reads as brightness.
Over ``--frames`` the collision plays out; a still catches it mid-merger. See
``docs/research/40-physics-sims.md``.
"""

import numpy as np

from ..engine import post
from ..scene import Scene
from ..sim.nbody import NBody, make_collision

_N = 4200
_DT = 0.011


def _render(width, height, time, mouse, device):
    n = _N
    if width * height <= 96 * 72:
        n = 700                                     # fast path for the smoke test

    pos, vel, mass, clump = make_collision(n=n, sep=2.7, radius=0.5,
                                           approach=0.36, spin=0.75, impact=0.42, seed=7)
    sim = NBody(pos, vel, mass, device=device, g=1.0, eps=0.055)
    steps = 205 + int(time * 55.0)                  # still (t=0) sits mid-merger with tidal tails
    pos, vel = sim.run(steps, _DT)

    speed = np.linalg.norm(vel, axis=1)
    sp = np.clip(speed / (np.percentile(speed, 95) + 1e-6), 0.0, 1.6)

    cool = np.array([0.45, 0.66, 1.0], np.float32)
    warm = np.array([1.0, 0.72, 0.4], np.float32)
    col = np.where(clump[:, None] == 0, cool[None, :], warm[None, :]).astype(np.float32)
    # fast particles whiten (shock-heated), giving the core its glare
    col = col + (np.array([1.0, 1.0, 1.0], np.float32)[None, :] - col) * (0.55 * sp)[:, None]
    bright = np.full(n, 0.5, np.float32)

    from ..sim.engine import splat_points
    ang = 0.25 + float(mouse[0]) * 0.004
    dist = 5.2
    eye = (dist * np.sin(ang), 1.5, dist * np.cos(ang))
    frame = splat_points(width, height, pos, col, bright, eye, (0.0, 0.0, 0.0),
                         fov_deg=40.0, stamp_radius=2)
    return post.tonemap(frame, mode="aces", exposure=1.15, preserve_hue=True)


SCENE = Scene(
    name="nbody",
    description="a real O(N²) N-body gravity simulation — two Plummer star clouds (cool-blue "
                "and warm-gold) colliding under Newtonian gravity on Warp, throwing curved "
                "tidal tails and merging into a relaxed cluster with a blazing dense core. "
                "Leapfrog integration, softened forces, additive particle splatting.",
    renderer=_render,
)
