"""A solenoid — a coil that makes a uniform magnetic field.

A current makes a magnetic field curling around the wire; wind the wire into a coil and
the loops' fields **add inside** to a nearly **uniform** field along the axis, while
outside they loop back like a bar magnet — an electromagnet from moving charge alone.
See ``docs/research/32-electromagnetism-and-fields.md``. --frames pushes the current.
"""

import math

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene


def _geometry(time):
    pts = []
    cols = []
    R = 0.85
    turns = 12
    # the copper coil (a helix), a current pulse travelling along it
    for s in np.linspace(0.0, 1.0, 900):
        ang = 2.0 * math.pi * turns * s
        z = -2.0 + 4.0 * s
        pulse = 0.5 + 0.5 * math.sin(ang - time * 6.0)
        c = np.array([1.0, 0.5, 0.2]) * (0.6 + 0.8 * pulse)
        pts.append([R * math.cos(ang), R * math.sin(ang), z]); cols.append(c)
    # uniform interior field lines (straight, along the axis)
    for (ox, oy) in [(0.0, 0.0), (0.4, 0.0), (-0.4, 0.0), (0.0, 0.4), (0.0, -0.4)]:
        for z in np.linspace(-1.9, 1.9, 60):
            pts.append([ox, oy, z]); cols.append([0.3, 0.6, 1.0])
    # exterior return loops (bulging back outside, like a bar magnet)
    for k in range(10):
        phi = 2.0 * math.pi * k / 10.0
        for s in np.linspace(0.0, math.pi, 46):
            rad = R + 1.4 * math.sin(s)
            z = 2.2 * math.cos(s)
            pts.append([rad * math.cos(phi), rad * math.sin(phi), z])
            cols.append([0.25, 0.45, 0.85])
    return np.array(pts, np.float32), np.array(cols, np.float32)


def _render(width, height, time, mouse, device):
    pts, cols = _geometry(time)
    hdr = splat_scene(pts, cols, width, height, time, device, foc=2.3, dist=6.0,
                      el=0.2, az0=0.6, az_speed=0.12, intensity=0.075,
                      bg=(0.01, 0.012, 0.02))
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="solenoid",
    description="A solenoid — a copper coil carrying current, its loops' fields adding to "
                "a uniform magnetic field straight along the axis inside and looping back "
                "outside like a bar magnet. --frames pushes the current along the coil.",
    renderer=_render,
)
