"""Process 7 — the chromosome: the telomere-capped fibre folds into a chromosome.

Chains from Process 6's actual output (the fibre with its two t-loop telomere caps). Two honest forms, both
selectable:

- **single chromatid** (``fold_chromosome``): one continuous strand condenses into a single rod — a
  centromere constriction at the middle, the two real **telomere** t-loops capping its two ends. Fully
  conserving: nothing is copied.
- **metaphase X** (``replicate_chromosome`` in ``replication.py``): the strand first **replicates** (the
  one place biology legitimately makes a copy — S-phase), then the two identical sister chromatids condense
  side by side, joined at the centromere — the classic X, four telomeres.

Conserving and physical: every base pair is folded (not regenerated) onto the rod, continuously; the
telomere caps are carried intact to the tips; nothing spawned (bar the explicit, shown replication in the
X form), nothing teleports. This lib supplies the two end states; ``scenes/warp_chromosome`` animates it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .telomere import cap_telomeres


@dataclasses.dataclass
class Chromosome:
    """The base pairs in two states: ``tel_a`` / ``tel_b`` (P,3) as Process 6 left them (fibre + t-loops),
    and ``chr_a`` / ``chr_b`` (P,3) folded into the single-chromatid chromosome. ``a_col`` / ``b_col`` the
    (telomere-tinted) base colours, ``is_tel`` (P,) telomeric pairs, ``arm_s`` (P,) position along the
    chromatid (0 = one telomere, 0.5 = centromere, 1 = the other telomere)."""

    tel_a: np.ndarray
    tel_b: np.ndarray
    chr_a: np.ndarray
    chr_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    is_tel: np.ndarray
    arm_s: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.tel_a.shape[0])


def fold_chromosome(sub: int = 2, block: int = 5, height: float = 6.5, rod: float = 1.5,
                    waist: float = 0.42) -> Chromosome:
    """Fold Process 6's telomere-capped strand into a single condensed chromatid — a rounded rod with a
    centromere constriction at the middle and the two real telomere t-loops capping the ends."""
    tl = cap_telomeres(sub=sub, block=block)
    p = tl.n_pairs
    i = np.arange(p)
    s = (i / (p - 1)).astype(np.float32)                 # 0 at end-0 telomere, 1 at end-1 telomere

    # the fibre's cross-section offset (its coil), so the folded rod stays round
    nx = 1
    # per-"fibre" mean would need the fibre grid; instead use a smooth local mean along the strand via a
    # block average so the rod cross-section is the coil radius, continuous along s.
    blk = 3960                                            # ~one fibre of base pairs
    fid = np.minimum(i // blk, (p - 1) // blk)
    mean_y = (np.bincount(fid, weights=tl.fib_a[:, 1]) / np.maximum(np.bincount(fid), 1))[fid]
    mean_z = (np.bincount(fid, weights=tl.fib_a[:, 2]) / np.maximum(np.bincount(fid), 1))[fid]
    perp_y = tl.fib_a[:, 1] - mean_y
    perp_z = tl.fib_a[:, 2] - mean_z

    # a straight vertical rod: telomere (top) → centromere waist → telomere (bottom)
    cy = height * (1.0 - 2.0 * s)
    taper = (waist + (1.0 - waist) * np.abs(np.sin(2.0 * np.pi * s))).astype(np.float32)
    scale = (rod * taper)[:, None]
    normal = np.array([1.0, 0.0, 0.0], np.float32)
    binorm = np.array([0.0, 0.0, 1.0], np.float32)
    centre = np.stack([np.zeros_like(cy), cy, np.zeros_like(cy)], 1).astype(np.float32)
    chr_a = (centre + scale * (perp_y[:, None] * normal + perp_z[:, None] * binorm)).astype(np.float32)
    perp_yb = tl.fib_b[:, 1] - mean_y
    perp_zb = tl.fib_b[:, 2] - mean_z
    chr_b = (centre + scale * (perp_yb[:, None] * normal + perp_zb[:, None] * binorm)).astype(np.float32)

    # keep the two telomere t-loops as loops, carried intact to the two rod tips (the caps)
    for end, mask, tip_y in ((0, i < tl.tel_len, height), (1, i >= p - tl.tel_len, -height)):
        m = mask
        loop = tl.tel_a[m]
        loop_off = tl.tel_b[m] - tl.tel_a[m]
        anchor = tl.ends[end]
        tip = np.array([0.0, tip_y, 0.0], np.float32)
        chr_a[m] = (loop - anchor + tip).astype(np.float32)
        chr_b[m] = (loop - anchor + tip + loop_off).astype(np.float32)

    return Chromosome(tel_a=tl.tel_a, tel_b=tl.tel_b, chr_a=chr_a, chr_b=chr_b,
                      a_col=tl.a_col.copy(), b_col=tl.b_col.copy(), is_tel=tl.is_tel, arm_s=s)
