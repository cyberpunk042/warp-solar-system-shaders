"""Quark — a single colour-charged quark (the up flavour by default).

A free quark can't be isolated (confinement), so it's one glowing plasma orb whose
QCD colour charge cycles red→green→blue over time, with gluon wisps radiating
outward (the field that would bind it). The six flavours are ``quark_up`` …
``quark_bottom`` (see ``quarks``). See ``docs/research/21-standard-model.md``.
iMouse orbits.
"""

from ..scene import Scene
from ..subatomic.quark import render_quark


def _render(width, height, time, mouse, device):
    return render_quark(width, height, time, mouse, device, flav=0)


SCENE = Scene(
    name="quark",
    description="A single quark — a colour-charge-cycling plasma orb with gluon "
                "wisps (the up flavour; confinement forbids isolating it). "
                "iMouse orbits.",
    renderer=_render,
)
