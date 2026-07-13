"""Cyclotron motion — a charged particle spiralling around a magnetic field.

A charged particle in a magnetic field feels a force perpendicular to its velocity
(**F = qv×B**), so it can't move straight: it circles around the field lines and, with
any drift along them, traces a **helix**. This is how particles are trapped in
magnetic bottles and radiation belts. See ``docs/research/32-electromagnetism-and-fields.md``.
--frames advances the particle.
"""

import math

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene


def _geometry(time):
    pts = []
    cols = []
    R = 0.72
    turns = 11
    zmin, zmax = -2.2, 2.2
    # straight magnetic field lines (the B field, along the axis)
    for (ox, oy) in [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0),
                     (0.8, 0.8), (-0.8, -0.8)]:
        for z in np.linspace(zmin, zmax, 44):
            pts.append([ox, oy, z]); cols.append([0.22, 0.4, 0.85])
    # the helical trajectory
    n = 900
    for m in range(n):
        s = m / (n - 1)
        ang = 2.0 * math.pi * turns * s
        z = zmin + (zmax - zmin) * s
        pts.append([R * math.cos(ang), R * math.sin(ang), z])
        cols.append([0.4, 0.85, 1.0])
    # the glowing particle at its current position along the helix
    sh = (time * 0.14) % 1.0
    angh = 2.0 * math.pi * turns * sh
    zh = zmin + (zmax - zmin) * sh
    ph = np.array([R * math.cos(angh), R * math.sin(angh), zh])
    rng = np.random.default_rng(1)
    for _ in range(260):
        d = rng.normal(size=3) * 0.05
        pts.append((ph + d).tolist()); cols.append([1.0, 0.9, 0.55])
    return np.array(pts, np.float32), np.array(cols, np.float32)


def _render(width, height, time, mouse, device):
    pts, cols = _geometry(time)
    hdr = splat_scene(pts, cols, width, height, time, device, foc=2.2, dist=6.2,
                      el=0.16, az0=0.62, az_speed=0.1, intensity=0.06,
                      bg=(0.01, 0.012, 0.02))
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.55, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.07)


SCENE = Scene(
    name="cyclotron",
    description="Cyclotron motion — a charged particle spiralling around magnetic field "
                "lines into a helix, forced sideways by F = qv×B, the bright particle "
                "riding its own trajectory. --frames advances the particle.",
    renderer=_render,
)
