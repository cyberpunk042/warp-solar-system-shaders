"""Process 3 — double helices: the base pairs wind into MANY short DNA double helices.

Chains from Process 2's actual output (the ordered base-pair field). The physical fact that fixes the
earlier mistake: a double helix only holds on the order of **~100 base pairs** (up to ~1000 at most) —
so 182872 base pairs do not make one giant helix, they make **many** short ones. The pairs are grouped
in sequence (``bp_per_helix`` each) and every group winds into its own clean, correctly-proportioned
double helix: ~10.5 base pairs per turn, and a pitch (rise per turn) about 3.4× the radius, the real
B-DNA geometry — so each helix is a slender, clearly-twisted thread, not a squashed cylinder. The result
is a whole **field of double helices**, one strand continuing into the next.

Conserving and physical: every base pair (every token) is placed in exactly one helix — nothing spawned,
nothing teleports. This lib supplies the geometry; ``scenes/warp_helix`` animates the gather + twist.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .basepair import bind_pairs

_BP_PER_TURN = 10.5      # real B-DNA: ~10.5 base pairs per helical turn
_PITCH_RATIO = 3.4       # real B-DNA: pitch (rise per turn) ≈ 3.4 × radius


@dataclasses.dataclass
class DoubleHelix:
    """The base pairs, grouped to wind into many short double helices. ``field_a``/``field_b`` (P,3) are
    the two tokens of each pair as Process 2 left them; ``a_col``/``b_col`` their base colours;
    ``centers`` (H,3) where each little helix stands (its axis vertical). ``bp_per_helix`` pairs per
    helix, ``radius`` / ``height`` / ``dtheta`` (twist per base pair) / ``groove`` parameterise each."""

    field_a: np.ndarray
    field_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    centers: np.ndarray
    bp_per_helix: int
    radius: float
    height: float
    dtheta: float
    groove: float

    @property
    def n_pairs(self) -> int:
        return int(self.field_a.shape[0])

    @property
    def n_helix(self) -> int:
        return int(self.centers.shape[0])


def wind_helix(sub: int = 2, block: int = 5, bp_per_helix: int = 110,
               radius: float = 0.30, groove: float = math.pi) -> DoubleHelix:
    """Group the Process-2 base pairs (``bp_per_helix`` each) into many short double helices with real
    B-DNA proportions. The helices stand on a grid that spans **the same footprint** Process 2's flat
    base-pair sheet occupied, so each strip of pairs winds up roughly **in place** (a gentle, physical
    gather — no flying across the room) into its own slender, well-spaced double helix: a whole field."""
    bp = bind_pairs(sub=sub, block=block)
    p = bp.n_pairs
    g = int(bp_per_helix)
    n_helix = (p + g - 1) // g

    turns = g / _BP_PER_TURN
    dtheta = 2.0 * math.pi / _BP_PER_TURN                # twist per base pair (fixed by physics)
    pitch = _PITCH_RATIO * radius                        # rise per turn ≈ 3.4 × radius
    height = turns * pitch                               # slender thread, not a squashed cylinder

    # grid spanning Process 2's own x/z footprint, so the helices stand where their pairs were: a sparse,
    # well-separated field (each helix isolated, so its twist reads), and the gather stays local.
    x0, x1 = float(bp.field_a[:, 0].min()), float(bp.field_a[:, 0].max())
    z0, z1 = float(bp.field_a[:, 2].min()), float(bp.field_a[:, 2].max())
    span_x, span_z = max(x1 - x0, 1e-3), max(z1 - z0, 1e-3)
    nx = max(int(round(math.sqrt(n_helix * span_x / span_z))), 1)
    nz = (n_helix + nx - 1) // nx
    sx, sz = span_x / max(nx - 1, 1), span_z / max(nz - 1, 1)
    gi = np.arange(n_helix)
    frac = lambda v: v - np.floor(v)
    jx = (frac(np.sin(gi * 12.9898) * 43758.5453) - 0.5) * 0.62 * sx
    jz = (frac(np.sin(gi * 78.2330 + 2.0) * 43758.5453) - 0.5) * 0.62 * sz
    cx = x0 + (gi % nx) * sx + jx
    cz = z0 + (gi // nx) * sz + jz
    centers = np.stack([cx, np.zeros_like(cx), cz], axis=1).astype(np.float32)

    return DoubleHelix(field_a=bp.field_a, field_b=bp.field_b, a_col=bp.a_col, b_col=bp.b_col,
                       centers=centers, bp_per_helix=g, radius=radius, height=height,
                       dtheta=dtheta, groove=groove)
