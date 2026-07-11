"""The stellar life-cycle — one star from birth to remnant, on a timeline.

`time` walks the normalized life ``t in [0, 1]`` over `_SPAN` seconds, so a single
``--time`` samples a phase and ``--frames`` plays the whole life:

    python render.py --scene stellar_lifecycle --time 10 -o giant.png     # mid-life
    python render.py --scene stellar_lifecycle --frames 120 --fps 6 --video life.mp4

Three initial masses show the **mass fork** (see
``docs/research/11-stellar-evolution.md``): a Sun-like star ends as a white dwarf,
a massive star as a neutron star, a very massive one as a black hole. Each frame
carries the H-R diagram inset.
"""

import numpy as np

from ..cosmos.stellar_evolution import render_lifecycle
from ..scene import Scene

_SPAN = 20.0                                  # seconds for one full life


def _life(mass):
    def _render(width, height, time, mouse, device):
        t = float(np.clip(time / _SPAN, 0.0, 1.0))
        return render_lifecycle(t, mass, width, height, device=device)
    return _render


SCENES = [
    Scene(name="stellar_lifecycle", renderer=_life(1.0),
          description="A Sun-like star's whole life: protostar -> main sequence -> "
                      "red giant -> planetary nebula -> white dwarf, with the H-R "
                      "inset. --frames over 20s plays it out."),
    Scene(name="stellar_massive", renderer=_life(14.0),
          description="A massive star (14 M_sun): O/B main sequence -> red "
                      "supergiant -> supernova -> neutron star (pulsar)."),
    Scene(name="stellar_collapse", renderer=_life(30.0),
          description="A very massive star (30 M_sun) ending in core-collapse: "
                      "red supergiant -> supernova -> black hole (lensed disk)."),
]
