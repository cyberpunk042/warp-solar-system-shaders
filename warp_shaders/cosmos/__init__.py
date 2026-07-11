"""cosmos — a configurable solar system built on the engine.

World-space celestial-body shaders (sun / neutron star / white dwarf / black
hole), Keplerian + N-body orbital mechanics, a positioned volumetric nebula, and
a layered depth-composited `SolarSystem` that places any mix of stars and
configurable planets on chosen orbits — the project's namesake.
"""

from .bodies import (BLACK_HOLE, NEUTRON, SUN, WHITE_DWARF, StarConfig,
                     make_star, render_star)

__all__ = ["StarConfig", "make_star", "render_star",
           "SUN", "NEUTRON", "WHITE_DWARF", "BLACK_HOLE"]
