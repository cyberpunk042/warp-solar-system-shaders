"""super-earth — a configurable procedural planet to stress-test the engine.

A "cheated" planet: every feature is an independent knob you can turn on or off —
atmosphere, ocean / lakes / rivers, mountains, volcanoes + lava, vegetation, a
living bioluminescence, snow, clouds, city lights — plus configurable **moons**
and a configurable **nuclear bombardment**. It renders a heightfield-displaced
sphere per pixel in a single Warp kernel driven by a :class:`PlanetConfig`, so the
same code path scales CPU→GPU on ``--quality``.

The point is to *play with the engine and find its limits* — a super-planet has
even more degrees of freedom (gas, storms) that build on this.
"""

from .planet import PlanetConfig, make_config, render_planet
from . import presets

__all__ = ["PlanetConfig", "make_config", "render_planet", "presets"]
