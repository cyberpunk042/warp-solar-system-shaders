"""Colliding galaxies — a Toomre restricted N-body encounter, as scenes.

`time` walks the fly-by over `_SPAN` seconds (`--frames` plays the whole
interaction). A **prograde** encounter throws long tidal tails + a bridge (the
Antennae look); a **retrograde** one barely responds — the classic Toomre &
Toomre 1972 contrast. See ``docs/research/12-galaxy-collisions.md``.

    python render.py --scene galaxy_collision --frames 64 --fps 6 --video out/tails.mp4
"""

import numpy as np

from ..cosmos.galaxy_dynamics import (EncounterConfig, GalaxyConfig,
                                      render_collision, simulate)
from ..scene import Scene

_SPAN = 12.0
_GOLD = (1.0, 0.82, 0.55)
_BLUE = (0.55, 0.72, 1.0)
_SIM_CACHE = {}


def _encounter(spin1, spin2, seed=4):
    g0 = GalaxyConfig(mass=1.0, n=2400, r_in=0.35, r_out=2.1, incl_deg=18, spin=spin1,
                      center=(-4.2, 0.0, 0.0), vel=(0.34, 0.0, 0.02), color=_GOLD)
    g1 = GalaxyConfig(mass=0.7, n=2000, r_in=0.35, r_out=1.8, incl_deg=35, spin=spin2,
                      center=(4.2, 1.5, 0.0), vel=(-0.34, 0.0, -0.02), color=_BLUE)
    return EncounterConfig(g0, g1, soft=0.28, seed=seed)


def _scene(key, enc, az=0.5, el=0.62, fov=40.0):
    def _render(width, height, time, mouse, device):
        sim = _SIM_CACHE.get(key)
        if sim is None:
            sim = simulate(enc, frames=64, substeps=10, dt=0.06)
            _SIM_CACHE[key] = sim
        f = int(round(float(np.clip(time / _SPAN, 0.0, 1.0)) * (sim.frames - 1)))
        return render_collision(sim, f, width, height,
                                az=az + float(mouse[0]) * 0.01,
                                el=el + float(mouse[1]) * 0.005, fov=fov)
    return _render


SCENES = [
    Scene(name="galaxy_collision", renderer=_scene("prograde", _encounter(1, 1)),
          description="Two galaxies in a prograde fly-by (Toomre restricted "
                      "N-body): tidal bridges + long tails unfurl. --frames to "
                      "play the encounter."),
    Scene(name="galaxy_retrograde", renderer=_scene("retro", _encounter(-1, -1)),
          description="The same encounter with retrograde disks — barely any tails "
                      "(the Toomre & Toomre 1972 prograde/retrograde contrast)."),
]
