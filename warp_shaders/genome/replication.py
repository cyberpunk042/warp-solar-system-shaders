"""Process 8 — replication → the metaphase chromosome (the X).

Chains from Process 7's single chromatid. To make the classic **X**, biology first **replicates** the DNA
(S-phase) — this is the one place a copy is legitimately made — producing two **identical sister
chromatids**. The sisters stay joined at the **centromere** and, at metaphase, tilt into the familiar X,
each keeping its two **telomere** caps (four telomeres in all).

This is the honest cost of the X: a copy is made (shown, not hidden — the two sisters begin coincident and
separate). Everything else is conserving — the sister is an exact copy of Process 7's chromatid, folded
the same way; the only new matter is the deliberate replication.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .chromosome import fold_chromosome


@dataclasses.dataclass
class Replication:
    """Both sister chromatids stacked (2P,3): ``single_a`` / ``single_b`` the pre-replication state (the
    two sisters coincident on Process 7's chromatid), ``x_a`` / ``x_b`` the metaphase X (the sisters tilted
    apart, joined at the centromere). ``a_col`` / ``b_col`` base colours, ``is_tel`` telomeric pairs."""

    single_a: np.ndarray
    single_b: np.ndarray
    x_a: np.ndarray
    x_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    is_tel: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.single_a.shape[0] // 2)


def _rot_z(pts, ang):
    c, s = math.cos(ang), math.sin(ang)
    x = pts[:, 0] * c - pts[:, 1] * s
    y = pts[:, 0] * s + pts[:, 1] * c
    return np.stack([x, y, pts[:, 2]], 1).astype(np.float32)


def replicate_chromosome(sub: int = 2, block: int = 5, tilt_deg: float = 13.0) -> Replication:
    """Replicate Process 7's chromatid into two identical sisters and tilt them into the metaphase X,
    joined at the centromere (the origin, y=0)."""
    cr = fold_chromosome(sub=sub, block=block)
    th = math.radians(tilt_deg)

    # sister A tilts one way, sister B the other, both about the centromere at the origin → the X
    a_a = _rot_z(cr.chr_a, +th)
    a_b = _rot_z(cr.chr_b, +th)
    b_a = _rot_z(cr.chr_a, -th)
    b_b = _rot_z(cr.chr_b, -th)
    x_a = np.concatenate([a_a, b_a]).astype(np.float32)
    x_b = np.concatenate([a_b, b_b]).astype(np.float32)

    # pre-replication: the two sisters coincident on the one chromatid (looks like a single chromatid)
    single_a = np.tile(cr.chr_a, (2, 1)).astype(np.float32)
    single_b = np.tile(cr.chr_b, (2, 1)).astype(np.float32)

    a_col = np.tile(cr.a_col, (2, 1)).astype(np.float32)
    b_col = np.tile(cr.b_col, (2, 1)).astype(np.float32)
    is_tel = np.tile(cr.is_tel, 2)

    return Replication(single_a=single_a, single_b=single_b, x_a=x_a, x_b=x_b,
                       a_col=a_col, b_col=b_col, is_tel=is_tel)
