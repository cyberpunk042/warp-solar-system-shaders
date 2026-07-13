"""Antimatter — the positron, the antiproton, and an annihilation event.

Every particle has an antiparticle with the same mass but opposite charge and
quantum numbers. Here: the **positron** (e⁺, Dirac's 1928 prediction, found by
Anderson 1932), the **antiproton** (p̄ = ū ū d̄, made at the Bevatron 1955), and
**e⁻ + e⁺ → γ γ** annihilation turning rest mass into two gamma photons. See
``docs/research/21-standard-model.md``.
"""

import functools

from ..scene import Scene
from ..subatomic.annihilation import render_annihilation
from ..subatomic.hadron import render_nucleon
from ..subatomic.lepton import render_lepton


def _positron(width, height, time, mouse, device):
    return render_lepton(width, height, time, mouse, device, kind=0, anti=True)


def _antiproton(width, height, time, mouse, device):
    return render_nucleon(width, height, time, mouse, device, is_proton=True, anti=True)


SCENES = [
    Scene(name="positron",
          description="the positron e⁺ (0.511 MeV) — the electron's antiparticle, a "
                      "bright core in a warm positive-charge field (charge-conjugated "
                      "Coulomb ripples). iMouse orbits.",
          renderer=_positron),
    Scene(name="antiproton",
          description="the antiproton p̄ (ū ū d̄) — three anti-colour antiquarks in a "
                      "violet confinement bag, the proton's mirror. iMouse orbits.",
          renderer=_antiproton),
    Scene(name="annihilation",
          description="electron–positron annihilation e⁻ + e⁺ → γ γ — matter meets "
                      "antimatter, flashes, and leaves as two back-to-back gamma "
                      "photons. --frames runs the event.",
          renderer=render_annihilation),
]
