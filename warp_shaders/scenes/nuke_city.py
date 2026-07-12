"""The nuke, tested on a city — a 300 kt strategic warhead over a downtown.

The [buildings](../buildings/) SDF kit meets [blast](../blast/) physics: a grid of
towers and office blocks that **collapse to rubble** as the overpressure front
sweeps out to the 5 psi destruction ring, under a physically-sized fireball and
rising mushroom cloud. Animate with ``--frames`` to watch the wave level the
skyline outward from ground zero. See ``docs/research/18-nuke-the-city.md``.
"""

from ..blast.render import render_city
from ..scene import Scene

_YIELD_KT = 300.0                       # a typical strategic city-buster


def _render(width, height, time, mouse, device):
    return render_city(width, height, time, mouse, device, _YIELD_KT)


SCENE = Scene(
    name="nuke_city",
    description="A 300 kt warhead over a city — SDF towers/blocks collapse to "
                "rubble as the overpressure front sweeps out, under a rising "
                "mushroom cloud (buildings kit + blast physics).",
    renderer=_render,
)
