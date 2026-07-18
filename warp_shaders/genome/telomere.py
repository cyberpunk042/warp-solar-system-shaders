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


def _tloop(u, anchor, outdir, stem_len, loop_radius, loop_turns):
    """A t-loop path in OPEN space beside the fibre. ``u`` in [0,1] runs anchor → free 3' tip. The terminal
    DNA first leaves the fibre along a short **stem** (outward, away from the packed forest, so it never
    threads the ropes), then arcs a **lasso** of radius ``loop_radius`` (``loop_turns`` around) in the
    vertical plane containing ``outdir``, and the free tip **tucks back inward** — the protective t-loop.
    Because the whole cap hangs outward into clear space, it can never pass through the fibre or itself
    (bar the real tip-into-duplex tuck)."""
    up = np.array([0.0, 1.0, 0.0], np.float32)
    e1 = outdir / max(np.linalg.norm(outdir), 1e-6)      # horizontal, outward from the forest
    e2 = up                                              # the lasso stands up in the (outward, vertical) plane

    us = 0.34                                            # first third: the straight stem leaving the fibre
    stem_end = anchor + e1 * stem_len                    # where the lasso begins
    centre = stem_end + e1 * loop_radius                 # lasso centre, further out; its rim touches stem_end

    t_stem = np.clip(u / us, 0.0, 1.0)[:, None]
    stem = anchor * (1.0 - t_stem) + stem_end * t_stem

    v = np.clip((u - us) / (1.0 - us), 0.0, 1.0)         # 0..1 around the lasso
    theta = math.pi + v * loop_turns * 2.0 * math.pi     # start at pi so the rim meets the stem end
    pull = (1.0 - 0.30 * v)[:, None]                     # the tip tucks inward (invades the duplex): the t-loop
    loop = centre + loop_radius * pull * (np.cos(theta)[:, None] * e1 + np.sin(theta)[:, None] * e2)

    return np.where((u < us)[:, None], stem, loop).astype(np.float32)


def cap_telomeres(sub: int = 2, block: int = 5, tel_frac: float = 0.004,
                  loop_radius: float = 2.1, loop_turns: float = 1.25,
                  stem_len: float = 2.4) -> Telomeres:
    """Curl Process 5's fibre-strand ends into two t-loop telomere caps. A linear strand has exactly two
    ends (pair 0 and pair P-1); each terminal stretch (bare duplex past the last nucleosome) leaves the
    fibre outward and lassoes a t-loop in open space."""
    fb = coil_fibre(sub=sub, block=block)
    p = fb.n_pairs
    tl = max(int(p * tel_frac), 32)                      # base pairs in each telomere

    tel_a = fb.fib_a.copy()
    tel_b = fb.fib_b.copy()
    is_tel = np.zeros(p, bool)
    off = (fb.fib_b - fb.fib_a)                           # keep the paired backbone offset through the loop

    cxz = np.array([fb.centers[:, 0].mean(), 0.0, fb.centers[:, 2].mean()], np.float32)   # forest centre

    def outward(anchor):
        d = np.array([anchor[0] - cxz[0], 0.0, anchor[2] - cxz[2]], np.float32)
        n = np.linalg.norm(d)
        return d / n if n > 1e-3 else np.array([1.0, 0.0, 0.0], np.float32)

    ends = []
    # end 0: pairs [0, tl); the free 3' tip is pair 0, the anchor is pair tl
    a0 = fb.fib_a[tl]
    u0 = (tl - np.arange(tl)).astype(np.float32) / float(tl)      # pair tl-1 → u≈0 (anchor), pair 0 → u≈1 (tip)
    tel_a[:tl] = _tloop(u0, a0, outward(a0), stem_len, loop_radius, loop_turns)
    tel_b[:tl] = tel_a[:tl] + off[:tl]
    is_tel[:tl] = True
    ends.append(a0)

    # end 1: pairs [p-tl, p); the free 3' tip is pair p-1, the anchor is pair p-tl-1
    a1 = fb.fib_a[p - tl - 1]
    u1 = (np.arange(p - tl, p) - (p - tl - 1)).astype(np.float32) / float(tl)   # p-tl → u≈0, p-1 → u≈1
    tel_a[p - tl:] = _tloop(u1, a1, outward(a1), stem_len, loop_radius, loop_turns)
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
