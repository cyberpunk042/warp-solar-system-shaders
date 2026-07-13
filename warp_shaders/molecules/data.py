"""Molecular geometries + CPK colours for the ball-and-stick scenes.

Positions in ångström-ish units (bond lengths from the CRC handbook), CPK
element colours. Each builder returns ``(atoms, bonds)`` where an atom is
``(pos3, radius, colour3)`` and a bond is ``(i, j)``. See
``docs/research/22-chemistry-and-molecules.md``.
"""

import math

# CPK element colours
H = (0.95, 0.95, 0.95)
C = (0.22, 0.22, 0.25)
O = (0.9, 0.12, 0.1)
N = (0.2, 0.32, 0.95)
CL = (0.3, 0.9, 0.35)
NA = (0.6, 0.35, 0.9)
S = (0.95, 0.85, 0.2)

# ball-and-stick radii (shrunk from van-der-Waals so bonds show)
_R = {"H": 0.3, "C": 0.42, "O": 0.42, "N": 0.42, "Cl": 0.52, "Na": 0.52}


def water():
    a = 52.25 * math.pi / 180.0                     # half of 104.5°
    dx, dy = 0.96 * math.sin(a), -0.96 * math.cos(a)
    atoms = [((0.0, 0.0, 0.0), _R["O"], O),
             ((dx, dy, 0.0), _R["H"], H),
             ((-dx, dy, 0.0), _R["H"], H)]
    return atoms, [(0, 1), (0, 2)]


def carbon_dioxide():
    atoms = [((0.0, 0.0, 0.0), _R["C"], C),
             ((1.16, 0.0, 0.0), _R["O"], O),
             ((-1.16, 0.0, 0.0), _R["O"], O)]
    return atoms, [(0, 1), (0, 2)]


def methane():
    d = 1.09 / math.sqrt(3.0)
    dirs = [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]
    atoms = [((0.0, 0.0, 0.0), _R["C"], C)]
    bonds = []
    for k, dd in enumerate(dirs):
        atoms.append((dd, _R["H"], H))
        bonds.append((0, k + 1))
    return atoms, bonds


def ammonia():
    atoms = [((0.0, 0.0, 0.0), _R["N"], N)]
    bonds = []
    for k in range(3):
        az = k * 2.0 * math.pi / 3.0
        atoms.append(((0.94 * math.cos(az), -0.38, 0.94 * math.sin(az)), _R["H"], H))
        bonds.append((0, k + 1))
    return atoms, bonds


def benzene():
    atoms = []
    bonds = []
    for k in range(6):
        ang = k * math.pi / 3.0
        atoms.append(((1.39 * math.cos(ang), 0.0, 1.39 * math.sin(ang)), _R["C"], C))
    for k in range(6):
        ang = k * math.pi / 3.0
        atoms.append(((2.48 * math.cos(ang), 0.0, 2.48 * math.sin(ang)), _R["H"], H))
    for k in range(6):
        bonds.append((k, (k + 1) % 6))              # ring
        bonds.append((k, k + 6))                     # C–H
    return atoms, bonds


BUILDERS = {
    "water": water, "carbon_dioxide": carbon_dioxide, "methane": methane,
    "ammonia": ammonia, "benzene": benzene,
}
