"""Combustion — CH₄ + 2 O₂ → CO₂ + 2 H₂O, animated.

The methane and oxygen molecules drift together, collide in a **flash** of
released energy (exothermic), and the atoms recombine into carbon dioxide and
water that drift apart — conserving every atom. Loops; animate with ``--frames``.
See ``docs/research/22-chemistry-and-molecules.md``.
"""

import numpy as np

from ..molecules import data as _d
from ..molecules.render import render_molecule
from ..scene import Scene

_O = _d.O
_H = _d.H
_C = _d.C
_RO = _d._R["O"]


def _place(atoms, bonds, off):
    off = np.asarray(off, np.float32)
    na = [((a[0][0] + off[0], a[0][1] + off[1], a[0][2] + off[2]), a[1], a[2]) for a in atoms]
    return na, list(bonds)


def _combine(parts):
    atoms, bonds, base = [], [], 0
    for a, b in parts:
        atoms += a
        bonds += [(i + base, j + base) for (i, j) in b]
        base += len(a)
    return atoms, bonds


def _o2(off):
    atoms = [((-0.6, 0.0, 0.0), _RO, _O), ((0.6, 0.0, 0.0), _RO, _O)]
    return _place(atoms, [(0, 1)], off)


def _render(width, height, time, mouse, device, period=7.0):
    prog = (time % period) / period

    if prog < 0.46:                                   # reactants drift together
        s = prog / 0.46
        parts = [_place(*_d.methane(), (-3.2 * (1.0 - s), 0.0, 0.0)),
                 _o2((3.0 * (1.0 - s) + 0.2, 1.3, 0.0)),
                 _o2((3.0 * (1.0 - s) + 0.2, -1.3, 0.0))]
    elif prog < 0.56:                                 # the fireball moment (clustered)
        parts = [_place(*_d.methane(), (-0.2, 0.0, 0.0)),
                 _o2((0.4, 1.0, 0.0)), _o2((0.4, -1.0, 0.0))]
    else:                                             # products drift apart
        s = (prog - 0.56) / 0.44
        parts = [_place(*_d.carbon_dioxide(), (0.0, 0.0, -2.4 * s)),
                 _place(*_d.water(), (2.6 * s + 0.3, 1.0 * s, 1.4 * s)),
                 _place(*_d.water(), (-2.6 * s - 0.3, -1.0 * s, 1.4 * s))]

    atoms, bonds = _combine(parts)
    img = render_molecule(width, height, time, mouse, device, atoms, bonds,
                          dist=13.0, fov=42.0)

    # exothermic flash at the transition (screen-space additive burst)
    if 0.4 < prog < 0.64:
        f = max(0.0, 1.0 - abs(prog - 0.52) / 0.12)
        h, w, _ = img.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2)
        flash = (f * f) * np.exp(-r2 * 9.0)
        warm = np.array([1.0, 0.72, 0.38], np.float32)
        img = np.clip(img + flash[..., None] * warm * 1.05, 0.0, 1.0)
    return img.astype(np.float32)


SCENE = Scene(
    name="combustion",
    description="Combustion CH₄ + 2O₂ → CO₂ + 2H₂O — the molecules drift together, "
                "flash (exothermic), and recombine into CO₂ + water. --frames animates.",
    renderer=_render,
)
