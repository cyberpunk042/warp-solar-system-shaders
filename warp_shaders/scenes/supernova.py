"""Supernova — a core-collapse explosion and its expanding shockwave.

The blinding flash, then a shock-heated ejecta shell (incandescent orange-red
behind a blue-white leading shock, Rayleigh-Taylor filaments) expanding and cooling
over the hot remnant core. Animate with ``--frames``. See
``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

from ..cosmos.supernova import render_supernova
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_supernova(width, height, time, mouse, device)


SCENE = Scene(
    name="supernova",
    description="A core-collapse supernova — flash, then an expanding shock-heated "
                "ejecta shell (orange-red ejecta, blue-white shock, RT filaments) "
                "cooling over the hot remnant core.",
    renderer=_render,
)
