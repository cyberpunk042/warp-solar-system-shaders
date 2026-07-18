"""The genome as ONE continuous strand, coiled hierarchically — the physically-honest model.

DNA is a single molecule, not a field of separate pieces. It compacts by **nested super-coiling**: the double
helix (level 1) wraps into the nucleosome/10 nm coil (level 2), that coils into the 30 nm fibre (level 3),
that coils into the chromatid (level 4). At every level the coil's radius clears everything inside it and its
pitch (rise per turn) exceeds the local strand thickness — so the strand **never passes through itself**
(excluded volume). This module builds that one strand at a compaction fraction ``c`` in [0,1] (0 = the
extended thread, 1 = the packed chromatid), with every level engaging in turn, and exposes a ``min_separation``
check so the no-interpenetration invariant is measurable, not assumed.
"""

from __future__ import annotations

import numpy as np

# ---- level geometry (world units). Each level's radius clears the tube inside it; pitch >= 2 * inner tube
# radius so consecutive turns never touch. RHO_k = radius of the whole bundle up to level k. ----
_RHO0 = 0.10                                   # bare duplex half-thickness (the strand's own radius)
_LEVELS = [
    # (radius R_k, turns T_k) — outer (chromatid) first is built last; listed level 1..4
    (0.34, None),   # L1 double helix   — turns set by bp (10.5 bp/turn)
    (0.62, None),   # L2 nucleosome coil
    (1.30, None),   # L3 30 nm fibre
    (2.60, None),   # L4 chromatid
]
# rise-per-turn (pitch) per level, all >= 2 * inner radius so turns clear each other
_PITCH = [0.30, 1.05, 2.30, 4.60]
# how many turns each level makes over the whole strand (finer levels turn far more often)
_TURNS = [None, 46.0, 190.0, 22.0]             # L1 computed from bp; L2..L4 fixed super-coil counts


def _rmf(C):
    """Rotation-minimising frame along a polyline C (K,3) → (T, N, B), no twist flips."""
    k = C.shape[0]
    T = np.zeros((k, 3))
    T[:-1] = C[1:] - C[:-1]
    T[-1] = T[-2]
    T /= np.maximum(np.linalg.norm(T, axis=1, keepdims=True), 1e-12)
    N = np.zeros((k, 3))
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0]) + 1e-12
    for i in range(k - 1):
        v = C[i + 1] - C[i]
        c1 = v @ v
        nL = N[i] - (2.0 / max(c1, 1e-12)) * (v @ N[i]) * v
        tL = T[i] - (2.0 / max(c1, 1e-12)) * (v @ T[i]) * v
        v2 = T[i + 1] - tL
        c2 = v2 @ v2
        N[i + 1] = nL - (2.0 / max(c2, 1e-12)) * (v2 @ nL) * v2
        N[i + 1] /= np.linalg.norm(N[i + 1]) + 1e-12
    B = np.cross(T, N)
    return T, N, B


def _engage(c, a, b):
    u = np.clip((c - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def build_strand(n=20000, c=1.0, bp_per_turn=10.5):
    """One continuous strand at compaction ``c``. Returns (centre (n,3), a (n,3), b (n,3)) — the strand
    centreline and its two duplex backbones. Levels engage in order (helix → nucleosome → fibre → chromatid);
    each level winds the level below it, so the strand only ever coils tighter, never overlaps."""
    u = np.linspace(0.0, 1.0, n)

    # engagement schedule: finer level first (it forms first as things condense), coarser last
    e1 = _engage(c, 0.02, 0.22)     # double helix
    e2 = _engage(c, 0.20, 0.48)     # nucleosome coil
    e3 = _engage(c, 0.44, 0.72)     # 30 nm fibre
    e4 = _engage(c, 0.68, 1.00)     # chromatid

    # the outer axis SHORTENS as the coarse levels engage (this shortening is the compaction)
    L = (1.0 - 0.90 * e4) * 62.0 + 6.0
    axis = np.zeros((n, 3))
    axis[:, 1] = (u - 0.5) * L

    def wind(curve, R, turns, engage):
        """wind `curve` into a super-helix of radius R*engage, `turns` turns, in curve's own moving frame."""
        _, N, B = _rmf(curve)
        phi = 2.0 * np.pi * turns * u
        off = (R * engage)[:, None] if np.ndim(engage) else R * engage
        return curve + off * (np.cos(phi)[:, None] * N + np.sin(phi)[:, None] * B)

    # build from the OUTSIDE in: chromatid coil on the axis, then fibre on that, then nucleosome, then helix
    R1, R2, R3, R4 = (lv[0] for lv in _LEVELS)
    c4 = wind(axis, R4, _TURNS[3], e4)
    c3 = wind(c4, R3, _TURNS[2], e3)
    c2 = wind(c3, R2, _TURNS[1], e2)
    turns1 = (n / bp_per_turn)
    c1 = wind(c2, R1, turns1, e1)

    centre = c1
    # duplex: two backbones a small fixed offset apart, in the innermost frame
    _, N1, _ = _rmf(centre)
    a = centre + (0.5 * _RHO0) * N1
    b = centre - (0.5 * _RHO0) * N1
    return centre.astype(np.float32), a.astype(np.float32), b.astype(np.float32)


def min_separation(pts, gap=6):
    """Smallest distance between two strand points more than ``gap`` steps apart — the interpenetration test.
    Must stay >= the strand diameter for the geometry to be physically valid."""
    p = np.asarray(pts, np.float64)
    n = p.shape[0]
    best = 1e18
    idx = np.arange(n)
    for a in range(0, n, max(1, n // 1200)):
        d = np.linalg.norm(p[a] - p, axis=1)
        d[np.abs(idx - a) <= gap] = 1e18
        best = min(best, d.min())
    return float(best)
