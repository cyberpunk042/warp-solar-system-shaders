"""Process 3 — the double helix: wind the base pairs into DNA.

A separate conserving process. Its INPUT is the base-pair field from Process 2. It assembles those pairs
into the **DNA double helix**: pair i becomes the i-th **rung**, and the pair's two tokens become the two
points on the two **backbones** that spiral around the axis. Reading the pairs in sequence and stepping
the twist angle + rise per pair traces the classic right-handed double helix (~10.5 base pairs per turn).

Conserving and physical: every base pair (and so every token) is placed exactly once — nothing spawned.
The rungs are the 182872 base pairs; the two backbones are those same pairs' tokens, in order. The strand
is very long (that length is exactly why the next steps coil it into a chromosome). This process stops at
the double helix.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .basepair import bind_pairs

_BACKBONE = np.array([0.72, 0.80, 0.95], np.float32)   # sugar-phosphate backbone (both strands alike)


@dataclasses.dataclass
class DoubleHelix:
    """The base pairs wound into DNA. ``s1``/``s2`` (P,3) are the two backbone points of each rung;
    ``rung_a``/``rung_b`` (P,3) the base colours at the two ends. Conserved: P rungs for P base pairs,
    the two backbones traced by the pairs' own tokens — none spawned."""

    s1: np.ndarray
    s2: np.ndarray
    rung_a: np.ndarray
    rung_b: np.ndarray
    backbone: np.ndarray
    axis_len: float

    @property
    def n_pairs(self) -> int:
        return int(self.s1.shape[0])


def wind_helix(sub: int = 2, block: int = 5, bp_per_turn: float = 10.5,
               radius: float = 0.55, rise: float = 0.075, groove: float = 2.4) -> DoubleHelix:
    """Wind the Process-2 base pairs into a right-handed double helix. Returns a :class:`DoubleHelix`."""
    bp = bind_pairs(sub=sub, block=block)
    p = bp.n_pairs
    i = np.arange(p, dtype=np.float64)

    theta = i * (2.0 * np.pi / bp_per_turn)     # twist per base pair -> ~10.5 bp/turn
    y = i * rise                                # rise per base pair (the strand climbs)

    s1 = np.stack([radius * np.cos(theta), y, radius * np.sin(theta)], axis=1)
    # second strand offset by the groove angle -> the characteristic major/minor grooves of B-DNA
    s2 = np.stack([radius * np.cos(theta + groove), y, radius * np.sin(theta + groove)], axis=1)

    return DoubleHelix(
        s1=s1.astype(np.float32), s2=s2.astype(np.float32),
        rung_a=bp.a_col, rung_b=bp.b_col,
        backbone=_BACKBONE.copy(), axis_len=float(y[-1] if p else 0.0),
    )
