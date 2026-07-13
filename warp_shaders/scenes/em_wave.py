"""An electromagnetic wave — E and B sustaining each other through space.

A changing electric field makes a magnetic field and vice-versa, so the two propagate
together as a wave: the **electric field** (red) and **magnetic field** (blue)
oscillate in phase, each perpendicular to the other and to the direction of travel,
moving at *c*. Light is this wave. See ``docs/research/32-electromagnetism-and-fields.md``.
--frames propagates the wave.
"""

import math

import numpy as np

from ..engine import post
from ..mathviz.splat import splat_scene
from ..scene import Scene


def _wave(time):
    pts = []
    cols = []
    k = 3.4
    w = 2.5
    A = 1.15
    # vector "combs": stems from the axis out to each field's amplitude
    for z in np.linspace(-3.0, 3.0, 200):
        e = A * math.sin(k * z - w * time)
        for f in np.linspace(0.0, 1.0, 7):
            pts.append([e * f, 0.0, z]); cols.append([1.0, 0.25, 0.15])
            pts.append([0.0, e * f, z]); cols.append([0.2, 0.5, 1.0])
    # bright field curves
    for z in np.linspace(-3.0, 3.0, 620):
        e = A * math.sin(k * z - w * time)
        pts.append([e, 0.0, z]); cols.append([1.0, 0.4, 0.25])
        pts.append([0.0, e, z]); cols.append([0.35, 0.6, 1.0])
    # the propagation axis
    for z in np.linspace(-3.0, 3.0, 260):
        pts.append([0.0, 0.0, z]); cols.append([0.5, 0.5, 0.6])
    return np.array(pts, np.float32), np.array(cols, np.float32)


def _render(width, height, time, mouse, device):
    pts, cols = _wave(time)
    hdr = splat_scene(pts, cols, width, height, time, device, foc=2.3, dist=6.8,
                      el=0.26, az0=0.72, az_speed=0.03, intensity=0.05,
                      bg=(0.01, 0.012, 0.02))
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.08)


SCENE = Scene(
    name="em_wave",
    description="An electromagnetic wave — the electric field (red) and magnetic field "
                "(blue) oscillating in phase, each perpendicular to the other and to the "
                "direction of travel, propagating at c. Light itself. --frames "
                "propagates the wave.",
    renderer=_render,
)
