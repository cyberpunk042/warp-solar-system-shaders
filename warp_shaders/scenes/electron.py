"""Electron — the lightest charged lepton, as a point charge in its EM field.

A free electron is point-like: a bright core wrapped in its animated
electromagnetic field (radial filaments + outgoing Coulomb ripples), the gen-I
cyan of the lepton family. (Bound in an atom it spreads into a probability cloud —
see the ``atom`` scene.) See ``docs/research/21-standard-model.md``. iMouse orbits.
"""

from ..scene import Scene
from ..subatomic.lepton import render_lepton


def _render(width, height, time, mouse, device):
    return render_lepton(width, height, time, mouse, device, kind=0)


SCENE = Scene(
    name="electron",
    description="The electron (0.511 MeV) — a point charge with a bright core in "
                "an animated cyan EM field (radial filaments + Coulomb ripples). "
                "iMouse orbits.",
    renderer=_render,
)
