"""The nuke, tested on a suburb — a neighbourhood of houses under the blast.

The human-scale counterpart to `nuke_city`: the [buildings](../buildings/)
`suburb_de` houses (pitched roofs, plaster walls) **collapse into burning rubble**
as the overpressure front sweeps out to the 5 psi ring — a smaller yield levels a
whole neighbourhood. Same collapse model and burning-city render as `nuke_city`,
just aimed at houses. See ``docs/research/18-nuke-the-city.md``.
"""

from ..blast.render import render_suburb
from ..scene import Scene

_YIELD_KT = 45.0                        # a smaller device — still flattens the block


def _render(width, height, time, mouse, device):
    return render_suburb(width, height, time, mouse, device, _YIELD_KT)


SCENE = Scene(
    name="nuke_suburb",
    description="A ~45 kt burst over a suburb — SDF houses collapse into a burning "
                "field of rubble as the overpressure front sweeps out, under a "
                "rising mushroom (buildings kit + blast physics).",
    renderer=_render,
)
