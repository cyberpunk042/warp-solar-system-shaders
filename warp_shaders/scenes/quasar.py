"""Quasar — a supermassive black hole firing twin relativistic jets.

The lensed black hole + Doppler-beamed accretion disk of the `black_hole` scene,
now with **relativistic bipolar jets**: collimated synchrotron beams along the
spin axis, shock knots drifting outward, the approaching jet beamed brighter — an
active galactic nucleus. See ``docs/research/19-extraordinary-cosmos.md``.
"""

from ..cosmos.quasar import render_quasar
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_quasar(width, height, time, mouse, device)


SCENE = Scene(
    name="quasar",
    description="A supermassive black hole with twin relativistic jets — lensed "
                "horizon, Doppler-beamed disk, collimated synchrotron beams with "
                "drifting shock knots (an active galactic nucleus).",
    renderer=_render,
)
