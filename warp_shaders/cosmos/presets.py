"""Named solar systems — the point of the feature: the same renderer draws a
neutron-star system, a binary, a trinary of mixed star types, a black-hole
system, or a system cradled in a nebula, by config alone. Planets carry their
own super-earth `PlanetConfig`, so each world is configurable too.
"""

from __future__ import annotations

import math

from ..superearth.planet import make_config
from .bodies import BLACK_HOLE, NEUTRON, SUN, WHITE_DWARF, make_star
from .orbits import Orbit
from .system import Nebula, Planet, Star, SystemConfig


def _earth(seed=1.0):
    return make_config(seed=seed, mountain=0.6, has_ocean=1, has_rivers=1,
                       snow=0.9, has_atmo=1, atmo=1.0, veg=0.9, cloud=0.5)


def _gas(seed=4.0):
    return make_config(seed=seed, has_ocean=0, has_atmo=0, snow=0.0, gas=1.0,
                       storm=0.3, spin=0.12)


def first():
    """The first system: a live, precessing neutron star + one planet on a
    chosen inclined elliptical orbit."""
    ns = make_star(kind=NEUTRON, radius=0.5, temp=0.9, activity=0.8, spin=2.0,
                   precess=0.4, seed=2.0)
    return SystemConfig(
        stars=[Star(cfg=ns, orbit=Orbit(period=0.0))],
        planets=[Planet(cfg=_earth(1.0),
                        orbit=Orbit(a=7.0, e=0.35, incl=0.15, period=8.0,
                                    phase=1.2), radius=0.9)],
        dist=16.0, az=0.6, el=0.2, fov=40.0)


def binary():
    """Two suns orbiting their barycentre + an earth-like planet around the
    pair."""
    s1 = make_star(kind=SUN, radius=1.0, temp=0.55, activity=0.7, spin=1.0, seed=1.0)
    s2 = make_star(kind=SUN, radius=0.8, temp=0.35, activity=0.6, spin=1.0, seed=5.0)
    return SystemConfig(
        stars=[Star(cfg=s1, orbit=Orbit(a=2.2, e=0.1, period=5.0, phase=0.0)),
               Star(cfg=s2, orbit=Orbit(a=2.75, e=0.1, period=5.0, phase=math.pi))],
        planets=[Planet(cfg=_earth(3.0),
                        orbit=Orbit(a=8.5, e=0.2, incl=0.1, period=12.0,
                                    phase=0.5), radius=0.8)],
        dist=20.0, az=0.6, el=0.24, fov=42.0)


def trinary():
    """Three stars of different kinds — a sun, a neutron star, a white dwarf —
    with a gas-giant planet on a wide orbit."""
    a = make_star(kind=SUN, radius=0.9, temp=0.55, activity=0.7, spin=1.0, seed=1.0)
    b = make_star(kind=NEUTRON, radius=0.4, temp=0.9, activity=0.8, spin=2.0,
                  precess=0.5, seed=2.0)
    c = make_star(kind=WHITE_DWARF, radius=0.45, temp=0.8, activity=0.3,
                  spin=0.5, seed=3.0)
    return SystemConfig(
        stars=[Star(cfg=a, orbit=Orbit(a=2.6, e=0.15, period=6.0, phase=0.0)),
               Star(cfg=b, orbit=Orbit(a=3.0, e=0.15, incl=0.3, period=6.0,
                                       phase=2.1)),
               Star(cfg=c, orbit=Orbit(a=3.4, e=0.15, incl=-0.2, period=6.0,
                                       phase=4.2))],
        planets=[Planet(cfg=_gas(4.0),
                        orbit=Orbit(a=9.5, e=0.1, incl=0.12, period=14.0,
                                    phase=1.0), radius=1.1)],
        dist=24.0, az=0.6, el=0.26, fov=44.0)


def blackhole():
    """A black hole at the centre, lensing a companion sun and an earth-like
    planet."""
    bh = make_star(kind=BLACK_HOLE, radius=1.0, activity=1.0, spin=1.0, seed=1.0)
    sun = make_star(kind=SUN, radius=1.0, temp=0.55, activity=0.7, spin=1.0, seed=2.0)
    return SystemConfig(
        stars=[Star(cfg=bh, orbit=Orbit(period=0.0)),
               Star(cfg=sun, orbit=Orbit(a=9.0, e=0.0, incl=0.05, period=10.0,
                                         phase=2.0))],
        planets=[Planet(cfg=_earth(7.0),
                        orbit=Orbit(a=6.0, e=0.1, incl=0.2, period=7.0,
                                    phase=3.5), radius=0.7)],
        dist=18.0, az=0.6, el=0.18, fov=40.0)


def nebula_cradle():
    """A sun with an earth-like world and a gas giant, cradled in a nebula."""
    sun = make_star(kind=SUN, radius=1.0, temp=0.5, activity=0.7, spin=1.0, seed=1.0)
    return SystemConfig(
        stars=[Star(cfg=sun, orbit=Orbit(period=0.0))],
        planets=[Planet(cfg=_earth(2.0),
                        orbit=Orbit(a=6.0, e=0.1, incl=0.1, period=8.0,
                                    phase=1.0), radius=0.7),
                 Planet(cfg=_gas(6.0),
                        orbit=Orbit(a=10.0, e=0.05, incl=-0.15, period=15.0,
                                    phase=3.0), radius=1.1)],
        nebula=Nebula(center=(0.0, 0.0, -2.0), radius=16.0, seed=3.0),
        dist=22.0, az=0.6, el=0.2, fov=46.0)


def collapse():
    """Destructive: two massive suns spiral in, merge, collapse to a black hole,
    which swallows the planet. Driven by the N-body dynamics (see ss_collapse)."""
    s1 = make_star(kind=SUN, radius=2.1, temp=0.5, activity=0.7, spin=1.0, seed=1.0)
    s2 = make_star(kind=SUN, radius=2.0, temp=0.45, activity=0.7, spin=1.0, seed=5.0)
    return SystemConfig(
        stars=[Star(cfg=s1, orbit=Orbit(a=2.5, e=0.05, period=6.0, phase=0.0)),
               Star(cfg=s2, orbit=Orbit(a=2.7, e=0.05, period=6.0, phase=math.pi))],
        planets=[Planet(cfg=_earth(4.0),
                        orbit=Orbit(a=4.2, e=0.1, period=8.0, phase=1.0),
                        radius=0.7)],
        scenario="destructive", dist=22.0, az=0.6, el=0.22, fov=42.0)


_REGISTRY = {
    "first": first,
    "binary": binary,
    "trinary": trinary,
    "blackhole": blackhole,
    "nebula_cradle": nebula_cradle,
    "collapse": collapse,
}


def get(name):
    return _REGISTRY[name]()


def names():
    return list(_REGISTRY)
