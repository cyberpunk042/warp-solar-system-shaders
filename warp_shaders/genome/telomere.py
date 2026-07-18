"""Process 6 — telomeres: the DNA strand's two ends fold into protective t-loop caps.

Chains from Process 5's actual output (the 30 nm fibre). The biology: the very ends of the linear DNA are
**telomeres** — long tandem repeats (TTAGGG…) whose single-stranded 3' overhang loops back and invades the
duplex, forming a **t-loop** (a lasso) that caps and protects the end. A linear strand has exactly **two**
ends, so there are exactly **two** telomeres — the terminal stretch at each end of the fibre curls into a
t-loop here, ready to sit at the chromosome's tips (Process 7).

Conserving and physical: only the terminal base pairs are reshaped (the strand curls back on itself); every
base pair is reused, nothing spawned, nothing teleports. This lib supplies the two end states (fibre → the
capped strand); ``scenes/warp_telomere`` animates the two ends curling into their t-loops.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .fibre import coil_fibre

_TELOMERE = np.array([0.45, 1.0, 0.55], np.float32)     # TTAGGG repeats — a distinct telomere green


@dataclasses.dataclass
class Telomeres:
    """The base pairs in two states: ``fib_a`` / ``fib_b`` (P,3) as Process 5 left them, and ``tel_a`` /
    ``tel_b`` (P,3) with the two ends curled into t-loops. ``a_col`` / ``b_col`` the (telomere-tinted)
    base colours, ``is_tel`` (P,) which pairs are telomeric, ``ends`` the two anchor points, ``tel_len``
    base pairs per telomere."""

    fib_a: np.ndarray
    fib_b: np.ndarray
    tel_a: np.ndarray
    tel_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    is_tel: np.ndarray
    ends: np.ndarray
    tel_len: int

    @property
    def n_pairs(self) -> int:
        return int(self.fib_a.shape[0])


def _tloop(u, anchor, tang, side, loop_radius, loop_turns):
    """A t-loop path: the strand leaves the anchor, arcs a lasso of radius ``loop_radius`` (``loop_turns``
    around) that hangs to the SIDE of the strand — tangent to the strand at the anchor, so the strand never
    passes through its own loop — and the free 3' tip tucks back beside the anchor. ``u`` in [0,1] runs
    anchor → free tip."""
    d1 = tang / max(np.linalg.norm(tang), 1e-6)          # outward, along the strand
    d2 = np.cross(d1, np.array([0.0, 0.0, 1.0], np.float32))
    if np.linalg.norm(d2) < 1e-4:                        # strand ~parallel to z → pick another perpendicular
        d2 = np.cross(d1, np.array([0.0, 1.0, 0.0], np.float32))
    d2 = d2 / max(np.linalg.norm(d2), 1e-6)
    sd2 = side * d2
    centre = anchor + sd2 * loop_radius                  # loop hangs to the side; its rim touches the anchor
    theta = -0.5 * math.pi + u * loop_turns * 2.0 * math.pi   # start at the anchor (nearest rim point)
    pull = (1.0 - 0.22 * u)[:, None]                     # the tip tucks slightly inward (the t-loop)
    return centre + loop_radius * pull * (np.cos(theta)[:, None] * d1 + np.sin(theta)[:, None] * sd2)


def cap_telomeres(sub: int = 2, block: int = 5, tel_frac: float = 0.016,
                  loop_radius: float = 1.9, loop_turns: float = 1.15) -> Telomeres:
    """Curl Process 5's fibre-strand ends into two t-loop telomere caps."""
    fb = coil_fibre(sub=sub, block=block)
    p = fb.n_pairs
    tl = max(int(p * tel_frac), 32)                      # base pairs in each telomere

    tel_a = fb.fib_a.copy()
    tel_b = fb.fib_b.copy()
    is_tel = np.zeros(p, bool)
    off = (fb.fib_b - fb.fib_a)                           # keep the paired backbone offset through the loop

    ends = []
    # end 0: pairs [0, tl); the free 3' tip is pair 0, the anchor is pair tl
    a0 = fb.fib_a[tl]
    tang0 = a0 - fb.fib_a[min(tl + 40, p - 1)]
    u0 = (tl - np.arange(tl)).astype(np.float32) / float(tl)      # pair tl-1 → u≈0 (anchor), pair 0 → u≈1 (tip)
    tel_a[:tl] = _tloop(u0, a0, tang0, +1.0, loop_radius, loop_turns)
    tel_b[:tl] = tel_a[:tl] + off[:tl]
    is_tel[:tl] = True
    ends.append(a0)

    # end 1: pairs [p-tl, p); the free 3' tip is pair p-1, the anchor is pair p-tl-1
    a1 = fb.fib_a[p - tl - 1]
    tang1 = a1 - fb.fib_a[max(p - tl - 41, 0)]
    u1 = (np.arange(p - tl, p) - (p - tl - 1)).astype(np.float32) / float(tl)   # p-tl → u≈0, p-1 → u≈1
    tel_a[p - tl:] = _tloop(u1, a1, tang1, -1.0, loop_radius, loop_turns)
    tel_b[p - tl:] = tel_a[p - tl:] + off[p - tl:]
    is_tel[p - tl:] = True
    ends.append(a1)

    a_col = fb.a_col.copy()
    b_col = fb.b_col.copy()
    a_col[is_tel] = _TELOMERE
    b_col[is_tel] = _TELOMERE

    return Telomeres(fib_a=fb.fib_a, fib_b=fb.fib_b, tel_a=tel_a.astype(np.float32),
                     tel_b=tel_b.astype(np.float32), a_col=a_col, b_col=b_col, is_tel=is_tel,
                     ends=np.array(ends, np.float32), tel_len=tl)
