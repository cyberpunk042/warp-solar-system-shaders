"""Moons — configurable number, type, and size, orbiting the super-earth.

A :class:`Moon` is a small orbiting body; :func:`moon_state` turns a list of them
plus a time into the flat arrays the planet kernel ray-tests. Types: ``rocky``,
``icy``, ``lava``, ``desert`` (each shaded differently in the kernel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TYPES = {"rocky": 0, "icy": 1, "lava": 2, "desert": 3}


@dataclass
class Moon:
    orbit: float = 3.0        # orbit radius (planet radius = 1)
    size: float = 0.27        # moon radius
    speed: float = 0.25       # orbital angular rate
    phase: float = 0.0        # starting angle
    incl: float = 0.3         # orbital-plane inclination (radians)
    kind: str = "rocky"


def moon_state(moons, time: float):
    """(positions Nx3, radii N, type-ids N) for `moons` at `time`."""
    n = len(moons)
    pos = np.zeros((max(n, 1), 3), np.float32)
    rad = np.zeros(max(n, 1), np.float32)
    typ = np.zeros(max(n, 1), np.int32)
    for i, m in enumerate(moons):
        ang = m.phase + m.speed * time
        ox = m.orbit * math.cos(ang)
        oz = m.orbit * math.sin(ang)
        pos[i] = (ox, oz * math.sin(m.incl), oz * math.cos(m.incl))
        rad[i] = m.size
        typ[i] = _TYPES.get(m.kind, 0)
    return pos, rad, typ


def moonset(name: str):
    """A few ready-made moon systems."""
    if name == "none":
        return []
    if name == "luna":
        return [Moon(orbit=3.2, size=0.27, speed=0.22, phase=0.6, incl=0.35)]
    if name == "twin":
        return [Moon(orbit=2.8, size=0.22, speed=0.30, phase=0.0, incl=0.2, kind="rocky"),
                Moon(orbit=3.9, size=0.30, speed=0.16, phase=2.4, incl=-0.4, kind="desert")]
    if name == "many":
        return [Moon(orbit=2.5, size=0.16, speed=0.40, phase=0.0, incl=0.15, kind="rocky"),
                Moon(orbit=3.1, size=0.24, speed=0.26, phase=1.4, incl=0.5, kind="icy"),
                Moon(orbit=3.8, size=0.19, speed=0.19, phase=3.0, incl=-0.3, kind="lava"),
                Moon(orbit=4.6, size=0.32, speed=0.13, phase=4.2, incl=0.25, kind="desert")]
    return []
