"""Kilonova — a neutron-star merger forging the heavy elements.

Two neutron stars chirp inward, merge in a flash, and light a kilonova: a fast
blue polar ejecta component and a slower red equatorial one (r-process glow), with
a brief short-gamma-ray-burst jet along the poles. Animate with ``--frames``. See
``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

from ..cosmos.kilonova import render_kilonova
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_kilonova(width, height, time, mouse, device)


SCENE = Scene(
    name="kilonova",
    description="A neutron-star merger — inspiral, merge flash, then two-colour "
                "r-process ejecta (blue poles, red equator) and a short-GRB jet.",
    renderer=_render,
)
