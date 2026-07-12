"""Gravitational waves — a chirping binary inspiral rippling spacetime.

Two compact bodies spiral together, radiating quadrupole gravitational waves that
warp the background starfield into concentric m=2 ripples, chirping upward until
they merge. Animate with ``--frames``. See
``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

from ..cosmos.gwaves import render_gwaves
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_gwaves(width, height, time, mouse, device)


SCENE = Scene(
    name="gravitational_waves",
    description="A chirping binary inspiral radiating gravitational waves — "
                "quadrupole ripples warp the starfield, rising in frequency until "
                "the two bodies merge.",
    renderer=_render,
)
