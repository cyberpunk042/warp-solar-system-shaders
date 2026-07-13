"""The baryons beyond the nucleon — the hyperons and the Δ resonance.

Swap strange quarks into the proton/neutron recipe and the **hyperon** family
appears: Λ, Σ, Ξ (the "cascade"), and the triply-strange Ω⁻ that the quark model
predicted before it was found (1964). Three up quarks give the Δ⁺⁺. Registered as
five scenes. See ``docs/research/21-standard-model.md``.
"""

import functools

from ..scene import Scene
from ..subatomic.baryon import render_baryon


def _render(name, width, height, time, mouse, device):
    return render_baryon(width, height, time, mouse, device, name=name)


_LABELS = {
    "lambda": "the Lambda Λ⁰ (u d s, 1116 MeV) — the lightest hyperon, one strange "
              "quark joining a proton's u d",
    "sigma":  "the Sigma Σ⁺ (u u s, 1189 MeV) — a charged strange baryon",
    "xi":     "the Xi Ξ⁰ (u s s, 1315 MeV) — the 'cascade', two strange quarks, "
              "decaying in a chain",
    "omega":  "the Omega Ω⁻ (s s s, 1672 MeV) — three strange quarks; predicted by "
              "the quark model in 1962 and found in 1964, clinching it",
    "delta":  "the Delta Δ⁺⁺ (u u u, 1232 MeV) — a spin-3/2 resonance whose three "
              "identical up quarks demanded a new quantum number: colour",
}

SCENES = [
    Scene(name=name, description=lbl + ". iMouse orbits.",
          renderer=functools.partial(_render, name))
    for name, lbl in _LABELS.items()
]
