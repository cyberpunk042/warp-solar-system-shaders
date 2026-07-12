"""The six leptons — electron, muon, tau and their three neutrinos.

Charged leptons (e, μ, τ) are bright cores wrapped in an animated EM field,
coloured by generation (cyan / green / violet) and sized by mass. Neutrinos
(νₑ, ν_μ, ν_τ) are near-invisible shimmers that oscillate between flavours.
Registered as six scenes. See ``docs/research/21-standard-model.md``.
"""

import functools

from ..scene import Scene
from ..subatomic.lepton import _LEP, render_lepton


def _render(kind, width, height, time, mouse, device):
    return render_lepton(width, height, time, mouse, device, kind=kind)


def _make():
    scenes = []
    labels = {
        0: "the electron (0.511 MeV) — the lightest charged lepton, a bright core "
           "in a cyan EM field",
        1: "the muon (105.7 MeV) — a heavier gen-II electron, green EM field",
        2: "the tau (1777 MeV) — the heavy gen-III lepton, violet EM field",
        3: "the electron neutrino — a near-invisible shimmer, oscillating flavour",
        4: "the muon neutrino — a faint oscillating shimmer",
        5: "the tau neutrino — a faint oscillating shimmer",
    }
    for kind, (name, mass, gen) in _LEP.items():
        if kind == 0:
            continue                                 # 'electron' registered in electron.py
        scenes.append(Scene(
            name=name,
            description=labels[kind] + ". iMouse orbits.",
            renderer=functools.partial(_render, kind),
        ))
    return scenes


SCENES = _make()
