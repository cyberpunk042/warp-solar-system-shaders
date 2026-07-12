"""Wormhole — an Ellis throat showing another universe, gravitationally lensed.

A sphere that isn't a sphere: rays missing the throat lens this universe's blue
nebula into an Einstein ring; rays entering it cross to a second, amber universe
fish-eyed across the disc; the exotic-matter rim glows. See
``docs/research/19-extraordinary-cosmos.md``. Orbit with ``--frames``.
"""

from ..cosmos.wormhole import render_wormhole
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_wormhole(width, height, time, mouse, device)


SCENE = Scene(
    name="wormhole",
    description="An Ellis wormhole — the throat lenses this universe into an "
                "Einstein ring and shows another universe through the portal, "
                "with an exotic-matter rim.",
    renderer=_render,
)
