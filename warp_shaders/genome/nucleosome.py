"""Process 4 — nucleosomes: the string of double helices beads up into "beads on a string".

Chains from Process 3's actual output (the wound double helices). The biology: a nucleosome is ~147 base
pairs of the double helix wrapped ~1.75 left-handed turns around a histone core, with ~50 base pairs of
**linker** DNA reaching to the next bead (a ~200 bp repeat). So a nucleosome is ~the size of **one** of
our double helices — the count barely changes (1663 helices → 1663 beads), the win is **spatial** (a long
helix wraps down into a compact bead). The count-funnel toward one chromosome comes later (Process 5's
30 nm fibre, Process 6's fold).

Conserving and physical: every base pair is reused — of each helix's 110 pairs, the middle ~70% **wrap**
around that bead's core and ~15% at each end are the **linker** to the neighbouring beads. Nothing is
spawned (the histone core is the empty centre the DNA wraps around, not added matter); nothing teleports.
This lib supplies the two end states (wound helix → bead); ``scenes/warp_nucleosome`` animates the wrap.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .helix import wind_helix, wound_positions


@dataclasses.dataclass
class Nucleosomes:
    """The base pairs in two states: ``helix_a`` / ``helix_b`` (P,3) as Process 3 wound them, and
    ``nuc_a`` / ``nuc_b`` (P,3) wrapped into beads-on-a-string. ``a_col`` / ``b_col`` their base colours,
    ``centers`` (H,3) the bead centres, ``bp_per_nuc`` base pairs per bead, ``wrap_turns`` the super-coil,
    ``core_radius`` the histone-core radius, ``link_frac`` the linker fraction at each end."""

    helix_a: np.ndarray
    helix_b: np.ndarray
    nuc_a: np.ndarray
    nuc_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    centers: np.ndarray
    bp_per_nuc: int
    wrap_turns: float
    core_radius: float
    link_frac: float
    grid_nx: int = 0
    grid_nz: int = 0

    @property
    def n_pairs(self) -> int:
        return int(self.helix_a.shape[0])

    @property
    def n_beads(self) -> int:
        return int(self.centers.shape[0])


def wrap_nucleosomes(sub: int = 2, block: int = 5, wrap_turns: float = 1.75,
                     core_radius: float = 0.42, bead_thick: float = 0.34,
                     dna_r: float = 0.05, link_frac: float = 0.16,
                     jitter: float = 0.06) -> Nucleosomes:
    """Wrap Process 3's wound double helices into nucleosome beads on a string. Each helix (its base pairs)
    becomes one bead: the middle stretch wraps ``wrap_turns`` around the core, the two ends are linker DNA
    reaching to the neighbouring beads.

    **The string is a single continuous local path.** The base pairs run 0..P in sequence, so consecutive
    beads must sit next to each other in space or the linker DNA between them would have to leap across the
    whole field (and cross every other bead — an interpenetration). The beads are therefore laid on a
    **serpentine (boustrophedon)** thread through the grid: each row runs opposite to the last, so bead
    ``k`` and bead ``k+1`` are always spatial neighbours and every linker is short and local. The bead cores
    are spaced well over their own diameter apart (~1.3 vs ~0.9), exactly as real "beads on a string" sit —
    separated by linker DNA — so the strand never passes through itself. Verified with
    ``genome.strand.min_separation``: the only sub-diameter approach left is a real one — the linker leaving
    a core brushing that core's own wrapped gyres, which touches in life too."""
    hx = wind_helix(sub=sub, block=block)
    helix_a, helix_b = wound_positions(hx)            # Process 3's actual end state — the chain input
    p = hx.n_pairs
    g = hx.bp_per_helix
    n = hx.n_helix
    nx = int(hx.grid_nx)

    # rebuild the grid footprint Process 3 used, then thread it as a serpentine so consecutive beads are
    # spatial neighbours (the "string" never jumps). Small deterministic jitter for an organic bead field,
    # bounded so cores keep clear of each other.
    c0 = hx.centers
    x0, x1 = float(c0[:, 0].min()), float(c0[:, 0].max())
    z0, z1 = float(c0[:, 2].min()), float(c0[:, 2].max())
    sx = (x1 - x0) / max(nx - 1, 1)
    sz = (z1 - z0) / max((n + nx - 1) // nx - 1, 1)
    bid = np.arange(n)
    row = bid // nx
    col = np.where(row % 2 == 0, bid % nx, nx - 1 - (bid % nx))   # boustrophedon: rows alternate direction
    frac = lambda v: v - np.floor(v)
    jx = (frac(np.sin(bid * 12.9898) * 43758.5453) - 0.5) * jitter * sx
    jz = (frac(np.sin(bid * 78.2330 + 2.0) * 43758.5453) - 0.5) * jitter * sz
    centers = np.stack([x0 + col * sx + jx, np.zeros(n), z0 + row * sz + jz], 1).astype(np.float32)

    i = np.arange(p)
    gi = i // g
    s = (i % g).astype(np.float32) / float(g)         # 0..1 along this bead's base pairs
    c = centers[gi]
    c_prev = centers[np.clip(gi - 1, 0, n - 1)]        # serpentine neighbours — spatially adjacent
    c_next = centers[np.clip(gi + 1, 0, n - 1)]

    lo, hi = link_frac, 1.0 - link_frac               # [lo,hi] wraps; [0,lo) and (hi,1] are linker

    # --- wrapped middle: a left-handed super-helix around the (vertical) core axis ---
    u = np.clip((s - lo) / max(hi - lo, 1e-6), 0.0, 1.0)      # 0..1 across the wrap
    phi = -u * wrap_turns * 2.0 * math.pi                     # left-handed
    axial = (u - 0.5) * bead_thick
    ox, oz = np.cos(phi), np.sin(phi)                         # outward radial of the wrap
    wrap = np.stack([c[:, 0] + core_radius * ox,
                     c[:, 1] + axial,
                     c[:, 2] + core_radius * oz], 1).astype(np.float32)
    out = np.stack([ox, np.zeros_like(ox), oz], 1).astype(np.float32)   # ribbon in/out edge direction

    # wrap entry (u=0) and exit (u=1) points, for the linker to meet
    entry = np.stack([c[:, 0] + core_radius, c[:, 1] - 0.5 * bead_thick, c[:, 2]], 1).astype(np.float32)
    exit_phi = -wrap_turns * 2.0 * math.pi
    exit_pt = np.stack([c[:, 0] + core_radius * math.cos(exit_phi),
                        c[:, 1] + 0.5 * bead_thick,
                        c[:, 2] + core_radius * math.sin(exit_phi)], 1).astype(np.float32)

    # --- linker ends: straight DNA from the mid-point with the neighbour bead into the wrap ---
    t_in = np.clip(s / max(lo, 1e-6), 0.0, 1.0)[:, None]
    lin_in = (0.5 * (c + c_prev)) * (1.0 - t_in) + entry * t_in
    t_out = np.clip((s - hi) / max(1.0 - hi, 1e-6), 0.0, 1.0)[:, None]
    lin_out = exit_pt * (1.0 - t_out) + (0.5 * (c + c_next)) * t_out

    base = wrap.copy()
    m_in = s < lo
    m_out = s > hi
    base[m_in] = lin_in[m_in]
    base[m_out] = lin_out[m_out]
    off = out * dna_r
    off[m_in] = 0.0
    off[m_out] = 0.0

    nuc_a = (base + off).astype(np.float32)
    nuc_b = (base - off).astype(np.float32)

    return Nucleosomes(helix_a=helix_a, helix_b=helix_b, nuc_a=nuc_a, nuc_b=nuc_b,
                       a_col=hx.a_col, b_col=hx.b_col, centers=centers, bp_per_nuc=g,
                       wrap_turns=wrap_turns, core_radius=core_radius, link_frac=link_frac,
                       grid_nx=hx.grid_nx, grid_nz=hx.grid_nz)
