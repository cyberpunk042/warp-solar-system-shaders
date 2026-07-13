"""Molecules — ball-and-stick renders of common molecules.

Water, carbon dioxide, methane, ammonia and benzene, sphere-traced as CPK-coloured
signed-distance fields with studio lighting. iMouse orbits. See
``docs/research/22-chemistry-and-molecules.md``.
"""

import functools

from ..molecules.data import BUILDERS
from ..molecules.render import render_molecule
from ..scene import Scene

_DESC = {
    "water": "Water (H₂O) — bent, 104.5° (red O, white H)",
    "carbon_dioxide": "Carbon dioxide (CO₂) — linear O=C=O",
    "methane": "Methane (CH₄) — tetrahedral, 109.5°",
    "ammonia": "Ammonia (NH₃) — trigonal pyramidal (blue N)",
    "benzene": "Benzene (C₆H₆) — the planar aromatic ring",
}


def _render(builder, width, height, time, mouse, device):
    atoms, bonds = builder()
    return render_molecule(width, height, time, mouse, device, atoms, bonds)


SCENES = [
    Scene(name=name, description=_DESC[name] + ". iMouse orbits.",
          renderer=functools.partial(_render, fn))
    for name, fn in BUILDERS.items()
]
