"""Tidal disruption event — a star spaghettified and devoured by a black hole.

A star wanders too close, is torn into a hot debris stream that winds into the
lensed black hole, and lights a brightening accretion flare as it is swallowed —
an animated event (``--frames``). See ``docs/research/19-extraordinary-cosmos.md``.
"""

from ..cosmos.tde import render_tde
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_tde(width, height, time, mouse, device)


SCENE = Scene(
    name="tidal_disruption",
    description="A tidal disruption event — a star torn into a hot spiral debris "
                "stream spiralling into a lensed black hole, with a brightening "
                "accretion flare (an animated event).",
    renderer=_render,
)
