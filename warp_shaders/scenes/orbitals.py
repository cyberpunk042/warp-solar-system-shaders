"""Hydrogen orbitals — the real shapes of the electron's standing waves.

Cycles through the true |ψ_{nlm}|² densities — **1s** sphere, **2s** shell, **2p**
dumbbell, **3p**, **3d z²** (torus + lobes), **3d cloverleaf** — each ray-marched
as a volumetric cloud with its actual nodes and lobes. Animate with ``--frames``
to walk the sequence; the camera orbits throughout. See
``docs/research/21-standard-model.md``.
"""

from ..scene import Scene
from ..subatomic.atom import render_named

# the sequence walked over time (one every ~2.6 s): 1s, 2p, 3d z², 3d cloverleaf, 2s, 3p
_SEQ = [0, 2, 4, 5, 1, 3]


def _render(width, height, time, mouse, device):
    idx = int(time / 2.6) % len(_SEQ)
    return render_named(width, height, time, mouse, device, orb=_SEQ[idx],
                        nucleus=0.6)


SCENE = Scene(
    name="orbitals",
    description="Hydrogen electron orbitals — the real |ψ_nlm|² shapes (1s, 2p, "
                "3d z², 3d cloverleaf, 2s, 3p) with their nodes and lobes, cycled "
                "over time. --frames walks the sequence.",
    renderer=_render,
)
