"""Process 4 — nucleosomes: coil the double helix into beads on a string.

A separate conserving process. Its INPUT is the double helix from Process 3. The strand is far too long
to stay extended, so it does its first level of packing: every ~200 base pairs, a stretch of ~146 wraps
~1.75 turns into a tight little super-coil — a **nucleosome bead** — joined to the next by a short
**linker**. The result is the classic *beads on a string*.

Conserving and physical: the beads are made **only of the DNA itself** — the strand wound tighter,
nothing added at the centre (no histone is spawned; we use what we transform). Every base pair is placed
exactly once; over time the extended strand draws in and winds into the beads (continuous motion). This
process stops at beads on a string — coiling that fibre further is the next process.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .basepair import bind_pairs

_PER = 200          # base pairs per nucleosome repeat (~146 wrapped + ~54 linker)
_WRAP = 146         # base pairs wrapped around the core
_TURNS = 1.75       # super-helical turns of the wrap
_RNUC = 0.72        # wrap radius (the bead)
_THICK = 0.42       # bead thickness along its axis
_SPACING = 2.7      # centre-to-centre bead spacing along the string
_DS = 0.019         # extended (un-wrapped) rise per base pair


@dataclasses.dataclass
class Nucleosomes:
    """The double helix coiled into beads on a string. ``extended`` (P,3) is the drawn-out strand;
    ``wrapped`` (P,3) the same base pairs wound into nucleosome beads; ``colors`` (P,3) per-base-pair
    colour. Conserved: P points for P base pairs, the beads made only of the strand — none spawned."""

    extended: np.ndarray
    wrapped: np.ndarray
    colors: np.ndarray
    n_beads: int
    centers: np.ndarray = None       # (n_beads,3) string-arranged bead centres
    bead_index: np.ndarray = None    # (P,) which bead each base pair belongs to

    @property
    def n_pairs(self) -> int:
        return int(self.extended.shape[0])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def wrap_nucleosomes(sub: int = 2, block: int = 5) -> Nucleosomes:
    """Coil the Process-3 double helix into nucleosome beads on a string. Returns :class:`Nucleosomes`."""
    bp = bind_pairs(sub=sub, block=block)
    colors = (0.5 * (bp.a_col + bp.b_col)).astype(np.float32)
    p = bp.n_pairs
    i = np.arange(p)
    n = i // _PER                                   # which nucleosome
    l = (i % _PER).astype(np.float64)              # position within the repeat
    n_beads = int(n.max()) + 1

    # --- extended strand: a long, gently waving line (the drawn-out DNA before packing)
    ex = i * _DS
    extended = np.stack([ex - ex.mean(),
                         0.30 * np.sin(i * 0.006),
                         0.30 * np.cos(i * 0.004)], axis=1).astype(np.float32)

    # --- per-nucleosome frame: centre on a meandering string, a tilted spool axis
    bn = np.arange(n_beads)
    cx = bn * _SPACING
    cx = cx - cx.mean()
    centre = np.stack([cx, 0.85 * np.sin(bn * 0.6), 0.85 * np.cos(bn * 0.37)], axis=1)
    axis = _unit(np.stack([0.4 * np.sin(bn * 1.1), np.ones_like(bn, float), 0.4 * np.cos(bn * 0.7)], axis=1))
    u = _unit(np.cross(axis, np.array([0.0, 0.0, 1.0])))
    v = _unit(np.cross(axis, u))

    # wrap end (exit) of each bead, and entry (a=0) of each bead — for the linker lerp
    a_end = _TURNS * 2.0 * np.pi
    entry = centre + _RNUC * u
    exit_ = (centre
             + _RNUC * (np.cos(a_end) * u + np.sin(a_end) * v)
             + axis * (0.5 * _THICK))

    # --- wrapped positions, per base pair
    cen_i = centre[n]
    u_i, v_i, ax_i = u[n], v[n], axis[n]
    a = (l / _WRAP) * a_end
    rise = (l / _WRAP - 0.5) * _THICK
    wrapped_pos = (cen_i
                   + _RNUC * (np.cos(a)[:, None] * u_i + np.sin(a)[:, None] * v_i)
                   + ax_i * rise[:, None])

    # linker: straight from this bead's exit to the next bead's entry
    nlink = np.clip(n + 1, 0, n_beads - 1)
    f = np.clip((l - _WRAP) / float(_PER - _WRAP), 0.0, 1.0)
    linker_pos = exit_[n] * (1.0 - f)[:, None] + entry[nlink] * f[:, None]

    is_link = (l >= _WRAP)[:, None]
    wrapped = np.where(is_link, linker_pos, wrapped_pos).astype(np.float32)

    return Nucleosomes(extended=extended, wrapped=wrapped, colors=colors, n_beads=n_beads,
                       centers=centre.astype(np.float32), bead_index=n.astype(np.int64))
