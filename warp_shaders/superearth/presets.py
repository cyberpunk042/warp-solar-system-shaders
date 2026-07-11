"""Named super-earth configurations — each turns a different set of knobs on.

Presets are the point: the same planet code renders a barren rock, an earth-like
world, a volcanic hell, a living world, an ocean world — by config alone. New
features (lava, vegetation, life, moons, nukes) light up as later phases land.
"""

from .planet import make_config


def barren():
    """A dead rock — mountains + craters, no water, no air, no life."""
    return make_config(seed=2.0, mountain=0.9, has_ocean=0, snow=0.0,
                       has_atmo=0, has_volcano=1, volcano_n=5, veg=0.0,
                       alive=0.0, cloud=0.0)


def earthlike():
    """Oceans, continents, mountains, snow caps, an atmosphere."""
    return make_config(seed=1.0, mountain=0.6, sea_level=0.0, has_ocean=1,
                       snow=1.0, has_atmo=1, atmo=1.0, veg=0.0, cloud=0.0)


def ocean_world():
    """Mostly water with scattered island arcs."""
    return make_config(seed=5.0, mountain=0.5, sea_level=0.35, has_ocean=1,
                       snow=0.7, has_atmo=1, atmo=1.1)


def volcanic():
    """A young molten world — many volcanoes (lava lands in the volcano phase)."""
    return make_config(seed=3.0, mountain=1.0, has_ocean=0, snow=0.0,
                       has_atmo=1, atmo=0.7, has_volcano=1, volcano_n=8,
                       lava=1.0)


_REGISTRY = {
    "barren": barren,
    "earthlike": earthlike,
    "ocean_world": ocean_world,
    "volcanic": volcanic,
}


def get(name):
    return _REGISTRY[name]()


def names():
    return list(_REGISTRY)
