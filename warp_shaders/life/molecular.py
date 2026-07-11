"""Molecular machines as ray-traced meshes — the bottom of the "show life" ladder.

Two generators that emit the same indexed-triangle :class:`~warp_shaders.life.mesh.Mesh`
the plants use, so they ray-cast through :func:`~warp_shaders.life.render.render_plant`
unchanged (call it with ``ground=False`` for a floating molecule):

- :func:`build_helix` — a **DNA double helix**: two anti-parallel sugar-phosphate
  backbone rails wound around a common axis, joined by colour-coded base-pair rungs.
  Geometry follows Watson & Crick (1953) B-DNA: ~10.5 bp per turn, ~3.4 Å rise, ~20 Å
  diameter (here in arbitrary units, that ratio preserved).
- :func:`build_protein` — a **polypeptide backbone** whose path interpolates from an
  extended chain to a compact fold (an α-helix flowing into a β-strand), coloured N→C
  along its length (the standard molecular-viz cue).

Both reuse the turtle :class:`~warp_shaders.life.turtle.Segment` as the tube primitive and
:func:`~warp_shaders.life.mesh.build_mesh` for tessellation.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .mesh import Mesh, build_mesh
from .turtle import Geometry, Segment

Vec3 = np.ndarray

# the four bases, each its own colour (A-T and G-C pair up)
_BASE_COLORS = {
    "A": np.array([0.90, 0.30, 0.30], np.float32),   # adenine  — red
    "T": np.array([0.95, 0.80, 0.25], np.float32),   # thymine  — yellow
    "G": np.array([0.30, 0.75, 0.40], np.float32),   # guanine  — green
    "C": np.array([0.35, 0.55, 0.95], np.float32),   # cytosine — blue
}
_COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}
_BACKBONE = np.array([0.78, 0.80, 0.86], np.float32)  # sugar-phosphate rail


def _tube(points: List[Vec3], radius: float, color: Vec3,
          out: List[Segment]) -> None:
    """Append a chain of equal-radius segments through `points` to `out`."""
    for a, b in zip(points[:-1], points[1:]):
        out.append(Segment(a.copy(), b.copy(), radius, radius, color.copy()))


def build_helix(bp: int = 16, radius: float = 1.0, rise: float = 0.34,
                bp_per_turn: float = 10.5, sub: int = 6, seed: int = 0,
                sides: int = 6) -> Tuple[Mesh, Tuple[Vec3, Vec3]]:
    """A DNA double helix of `bp` base pairs → ``(Mesh, (lo, hi))``.

    `sub` sub-samples each base-pair interval so the rails read as smooth tubes.
    Bases are drawn from a seeded sequence; the two rails are the strand and its
    complement, offset by π around the axis (B-DNA `bp_per_turn`, `rise`).
    """
    rng = np.random.default_rng(seed)
    bases = rng.choice(list("ATGC"), size=max(bp, 0))
    dtheta = 2.0 * math.pi / bp_per_turn

    def rail(offset: float, k: float) -> Vec3:            # k in base-pair units
        th = k * dtheta + offset
        return np.array([radius * math.cos(th), k * rise, radius * math.sin(th)],
                        np.float32)

    railA: List[Vec3] = []
    railB: List[Vec3] = []
    n = max(bp - 1, 0)
    steps = n * sub
    for s in range(steps + 1):
        k = s / float(sub)
        railA.append(rail(0.0, k))
        railB.append(rail(math.pi, k))

    segs: List[Segment] = []
    _tube(railA, radius * 0.14, _BACKBONE, segs)
    _tube(railB, radius * 0.14, _BACKBONE, segs)
    # base-pair rungs: two half-rungs meeting at the axis, coloured per base
    for i in range(bp):
        k = float(i)
        a = rail(0.0, k)
        b = rail(math.pi, k)
        mid = (a + b) * 0.5
        base = bases[i] if i < len(bases) else "A"
        segs.append(Segment(a.copy(), mid.copy(), radius * 0.09, radius * 0.09,
                            _BASE_COLORS[base].copy()))
        segs.append(Segment(mid.copy(), b.copy(), radius * 0.09, radius * 0.09,
                            _BASE_COLORS[_COMPLEMENT[base]].copy()))

    geo = Geometry(segments=segs)
    return build_mesh(geo, sides=sides, min_radius=0.002), geo.bounds()


def _ramp(t: float) -> Vec3:
    """N→C colour ramp (blue → cyan → green → yellow → red), t in [0,1]."""
    stops = np.array([[0.20, 0.30, 0.95], [0.20, 0.80, 0.85],
                      [0.30, 0.80, 0.35], [0.95, 0.80, 0.20],
                      [0.92, 0.25, 0.25]], np.float32)
    x = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = int(x); f = x - i
    if i >= len(stops) - 1:
        return stops[-1].copy()
    return (stops[i] * (1.0 - f) + stops[i + 1] * f).astype(np.float32)


def build_protein(n: int = 48, fold: float = 1.0, sides: int = 6
                  ) -> Tuple[Mesh, Tuple[Vec3, Vec3]]:
    """A polypeptide backbone folding from extended (`fold`=0) to compact (`fold`=1).

    The folded target is an α-helix (residues 0..~55%) flowing through a turn into
    an antiparallel β-strand running back alongside it — a minimal fold motif. The
    backbone tube is coloured N→C. Returns ``(Mesh, (lo, hi))``.
    """
    fold = min(max(fold, 0.0), 1.0)
    ext_step = 0.5                                     # extended chain spacing
    a_turn = math.radians(100.0)                       # α-helix ~100°/residue
    a_rad, a_rise = 0.9, 0.28
    split = int(n * 0.55)
    pts: List[Vec3] = []
    for k in range(n):
        ext = np.array([0.0, k * ext_step, 0.0], np.float32)
        if k < split:                                  # α-helix
            th = k * a_turn
            fold_p = np.array([a_rad * math.cos(th), k * a_rise,
                               a_rad * math.sin(th)], np.float32)
        else:                                          # turn + β-strand back
            j = k - split
            top = split * a_rise
            fold_p = np.array([a_rad * 0.6, top - j * a_rise * 0.8,
                               -a_rad * 0.9], np.float32)
        pts.append(ext * (1.0 - fold) + fold_p * fold)

    segs: List[Segment] = []
    for i in range(len(pts) - 1):
        c = _ramp(i / float(max(len(pts) - 2, 1)))
        segs.append(Segment(pts[i].copy(), pts[i + 1].copy(), 0.16, 0.16, c))
    geo = Geometry(segments=segs)
    return build_mesh(geo, sides=sides, min_radius=0.002), geo.bounds()
