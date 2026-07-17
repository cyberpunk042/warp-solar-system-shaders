"""Process 6 — the chromosome: the 30 nm fibres fold into the metaphase chromosome.

Chains from Process 5's actual output (the 30 nm fibres). The final packing: the fibres condense onto a
protein scaffold and fold into the two sister **chromatids** of the metaphase chromosome, joined at the
**centromere** (the primary constriction) and capped at their four tips by the **telomeres**. The ~47
fibres split between the two chromatids (~24 + ~23); within each chromatid the fibres lay head-to-tail
along a bowed centre-line (telomere → centromere waist → telomere), the fibre coil crushed along the axis
so it fills the condensed arm — the last ~40× of compaction, DNA finally a chromosome.

Conserving and physical: every base pair is reused — each fibre is folded (not regenerated) onto its arm,
continuously, nothing spawned, nothing teleports. This lib supplies the two end states (fibre band →
chromosome); ``scenes/warp_chromosome`` animates the fold.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .fibre import coil_fibre

# telomere / centromere tints (the DNA body keeps its A/T/G/C colours; these mark the landmarks)
_TELOMERE = np.array([0.35, 0.95, 1.0], np.float32)     # bright cyan caps at the four arm-tips
_CENTROMERE = np.array([1.0, 0.55, 0.35], np.float32)   # warm constriction where the chromatids join


@dataclasses.dataclass
class Chromosome:
    """The base pairs in two states: ``fib_a`` / ``fib_b`` (P,3) as Process 5 left them (the fibre band),
    and ``chr_a`` / ``chr_b`` (P,3) folded into the chromosome. ``a_col`` / ``b_col`` the (landmark-tinted)
    base colours, ``chromatid`` (P,) which sister chromatid each pair went to, ``arm_t`` (P,) its position
    along that chromatid (0 = one telomere, 0.5 = centromere, 1 = other telomere)."""

    fib_a: np.ndarray
    fib_b: np.ndarray
    chr_a: np.ndarray
    chr_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    chromatid: np.ndarray
    arm_t: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.fib_a.shape[0])


def _centreline(t, x_side, arm_x, height, waist):
    """Chromatid centre-line: bows out to the telomere tips (t=0,1) and pinches to the centromere waist
    (t=0.5). Returns position (…,3)."""
    bow = np.abs(1.0 - 2.0 * t)                         # 1 at the tips, 0 at the waist
    x = x_side * (waist + (arm_x - waist) * bow)
    y = height * (1.0 - 2.0 * t)
    z = np.zeros_like(t)
    return np.stack([x, y, z], -1)


def fold_chromosome(sub: int = 2, block: int = 5, arm_x: float = 2.6, height: float = 7.0,
                    waist: float = 0.45, rod: float = 1.15) -> Chromosome:
    """Fold Process 5's fibres into the two chromatids of the metaphase chromosome — fibres split between
    the sisters, laid head-to-tail along each bowed arm and crushed to fill the condensed rod."""
    fb = coil_fibre(sub=sub, block=block)
    g = int(fb.bp_per_bead)
    nx = int(fb.beads_per_fibre)
    nf = int(fb.n_fibres)
    p = fb.n_pairs

    i = np.arange(p)
    f = (i // g) // nx                                  # which fibre each base pair belongs to
    c = (f % 2).astype(np.int64)                        # sister chromatid (fibres alternate)
    r = (f // 2).astype(np.float32)                     # rank of the fibre within its chromatid
    k0 = (nf + 1) // 2                                  # fibres in chromatid 0
    k1 = nf // 2                                        # fibres in chromatid 1
    kc = np.where(c == 0, k0, k1).astype(np.float32)

    # position along the fibre axis (x is the shared, centred solenoid axis in Process 5)
    x0, x1 = float(fb.fib_a[:, 0].min()), float(fb.fib_a[:, 0].max())
    along = (fb.fib_a[:, 0] - x0) / max(x1 - x0, 1e-6)
    t = np.clip((r + along) / kc, 0.0, 1.0).astype(np.float32)

    # the fibre's cross-section offset (perpendicular to its axis), from the fibre's own mean
    mean_y = (np.bincount(f, weights=fb.fib_a[:, 1]) / np.maximum(np.bincount(f), 1))[f]
    mean_z = (np.bincount(f, weights=fb.fib_a[:, 2]) / np.maximum(np.bincount(f), 1))[f]
    perp_y = fb.fib_a[:, 1] - mean_y
    perp_z = fb.fib_a[:, 2] - mean_z

    x_side = np.where(c == 0, -1.0, 1.0).astype(np.float32)
    eps = 1e-3
    c0 = _centreline(t, x_side, arm_x, height, waist)
    c1 = _centreline(t + eps, x_side, arm_x, height, waist)
    tang = c1 - c0
    tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-6)
    binorm = np.tile(np.array([0.0, 0.0, 1.0], np.float32), (p, 1))
    normal = np.cross(binorm, tang)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-6)

    # rod radius: fat arms, tapering to the telomere tips and pinching at the centromere waist
    taper = (0.32 + 0.68 * np.abs(np.sin(2.0 * np.pi * t))).astype(np.float32)
    scale = (rod * taper)[:, None]
    chr_a = (c0 + scale * (perp_y[:, None] * normal + perp_z[:, None] * binorm)).astype(np.float32)

    # the b backbone: same fold applied to Process 5's b positions (keep the paired offset)
    perp_yb = fb.fib_b[:, 1] - mean_y
    perp_zb = fb.fib_b[:, 2] - mean_z
    chr_b = (c0 + scale * (perp_yb[:, None] * normal + perp_zb[:, None] * binorm)).astype(np.float32)

    # landmark tints: telomeres at the four tips, centromere at the waist (body keeps A/T/G/C colours)
    a_col = fb.a_col.copy()
    b_col = fb.b_col.copy()
    tip = (t < 0.045) | (t > 0.955)
    cen = np.abs(t - 0.5) < 0.02
    a_col[tip] = _TELOMERE; b_col[tip] = _TELOMERE
    a_col[cen] = _CENTROMERE; b_col[cen] = _CENTROMERE

    return Chromosome(fib_a=fb.fib_a, fib_b=fb.fib_b, chr_a=chr_a, chr_b=chr_b,
                      a_col=a_col, b_col=b_col, chromatid=c, arm_t=t)
