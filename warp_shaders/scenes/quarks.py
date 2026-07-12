"""The six quark flavours — up, down, charm, strange, top, bottom.

Each is a flavour-tinted, colour-charge-cycling plasma orb sized by log(mass):
the near-massless up/down through to the enormous top. Registered as six scenes
(``quark_up`` … ``quark_bottom``). See ``docs/research/21-standard-model.md``.
"""

import functools

from ..scene import Scene
from ..subatomic.quark import _FLAV, render_quark


def _make():
    scenes = []
    for flav, (name, mass) in _FLAV.items():
        scenes.append(Scene(
            name=f"quark_{name}",
            description=f"The {name} quark ({mass:g} MeV) — a flavour-tinted, "
                        f"colour-charge-cycling plasma orb (size ∝ log mass), with "
                        f"gluon wisps. iMouse orbits.",
            renderer=functools.partial(_render, flav),
        ))
    return scenes


def _render(flav, width, height, time, mouse, device):
    return render_quark(width, height, time, mouse, device, flav=flav)


SCENES = _make()
