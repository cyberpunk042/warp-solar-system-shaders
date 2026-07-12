"""Proton — three quarks (u, u, d) in a confinement bag, bound by gluon flux tubes.

High-quality volumetric render: each quark is a turbulent colour-charged plasma
(red/green/blue → colour-neutral), bound by textured, flowing **gluon flux tubes**
(the QCD colour string), all inside a warm confinement bag (the proton is +1).
iMouse orbits. See ``docs/research/21-standard-model.md``.
"""

from ..scene import Scene
from ..subatomic.hadron import render_nucleon


def _render(width, height, time, mouse, device):
    return render_nucleon(width, height, time, mouse, device, is_proton=True)


SCENE = Scene(
    name="proton",
    description="A proton (uud) — three colour-charged quark plasmas bound by "
                "flowing gluon flux tubes in a warm confinement bag, ray-marched "
                "as a volumetric field. iMouse orbits.",
    renderer=_render,
)
