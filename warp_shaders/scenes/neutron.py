"""Neutron — three quarks (u, d, d) in a confinement bag, bound by gluon flux tubes.

Same volumetric machinery as the proton with the udd flavour content and a cool
(charge-neutral) confinement tint. iMouse orbits. See
``docs/research/21-standard-model.md``.
"""

from ..scene import Scene
from ..subatomic.hadron import render_nucleon


def _render(width, height, time, mouse, device):
    return render_nucleon(width, height, time, mouse, device, is_proton=False)


SCENE = Scene(
    name="neutron",
    description="A neutron (udd) — three colour-charged quark plasmas bound by "
                "flowing gluon flux tubes in a cool (neutral) confinement bag, "
                "ray-marched as a volumetric field. iMouse orbits.",
    renderer=_render,
)
