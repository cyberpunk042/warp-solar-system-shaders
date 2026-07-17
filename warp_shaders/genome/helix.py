"""Process 3 — the double helix: wind the base pairs into DNA.

A conserving process that **chains from Process 2's actual output**. Its input is the ordered base-pair
field (``BasePairs.field_a`` / ``field_b`` — the unwound ladder of rungs the tokens settled onto). It
does not regenerate anything: the two tokens of every base pair are the two ends of a rung, and those
same tokens become the two backbones of the double helix.

The transform is a real, continuous winding: the field of rungs first gathers into a single straight
ladder (rungs stacked in sequence), then the ladder **twists** about its axis into the right-handed
double helix. Nothing is spawned, nothing teleports — every point moves continuously from where Process
2 left it to its place on the helix. This lib supplies the geometry; ``scenes/warp_helix`` animates the
gather + twist.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .basepair import bind_pairs


@dataclasses.dataclass
class DoubleHelix:
    """The base pairs, ready to wind into DNA. ``field_a``/``field_b`` (P,3) are the two tokens of each
    pair as Process 2 left them (the input); ``a_col``/``b_col`` their base colours. ``height``,
    ``radius``, ``dtheta`` (twist per base pair) and ``groove`` parameterise the target helix — the two
    backbones are the same tokens, so nothing is spawned."""

    field_a: np.ndarray
    field_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    height: float
    radius: float
    dtheta: float
    groove: float

    @property
    def n_pairs(self) -> int:
        return int(self.field_a.shape[0])


def wind_helix(sub: int = 2, block: int = 5, radius: float = 1.25, height: float = 6.8,
               turns: float = 18.0, groove: float = math.pi) -> DoubleHelix:
    """Prepare the Process-2 base pairs to wind into a double helix of ``turns`` full turns."""
    bp = bind_pairs(sub=sub, block=block)
    p = bp.n_pairs
    dtheta = turns * 2.0 * math.pi / float(p)          # twist per base pair, over the whole strand
    return DoubleHelix(field_a=bp.field_a, field_b=bp.field_b, a_col=bp.a_col, b_col=bp.b_col,
                       height=height, radius=radius, dtheta=dtheta, groove=groove)
