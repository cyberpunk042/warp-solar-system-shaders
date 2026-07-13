"""The mesons — quark–antiquark hadrons.

Four flavours of the two-body strong-force bound state: the light **pion** (u d̄,
the lightest hadron, the pion that binds nuclei), the strange **kaon** (u s̄), and
the heavy quarkonia **J/ψ** (c c̄) and **Υ** (b b̄) — a charm and a bottom quark
orbiting their own antiquarks. Each is a colour+anti-colour pair on one gluon flux
string. Registered as four scenes. See ``docs/research/21-standard-model.md``.
"""

import functools

from ..scene import Scene
from ..subatomic.meson import render_meson


def _render(name, width, height, time, mouse, device):
    return render_meson(width, height, time, mouse, device, name=name)


_LABELS = {
    "pion":    "the pion π⁺ (u d̄, 140 MeV) — the lightest meson, the strong-force "
               "glue between nucleons",
    "kaon":    "the kaon K⁺ (u s̄, 494 MeV) — a strange meson, its s̄ quark makes it "
               "long-lived",
    "jpsi":    "the J/ψ (c c̄, 3.1 GeV) — charmonium, a charm quark bound to its "
               "antiquark; its 1974 discovery confirmed the charm quark",
    "upsilon": "the Υ upsilon (b b̄, 9.46 GeV) — bottomonium, a tightly bound bottom "
               "quark–antiquark pair",
}

SCENES = [
    Scene(name=name, description=lbl + ". iMouse orbits.",
          renderer=functools.partial(_render, name))
    for name, lbl in _LABELS.items()
]
