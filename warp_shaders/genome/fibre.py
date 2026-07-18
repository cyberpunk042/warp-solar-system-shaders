"""Process 5 — the 30 nm fibre: the beads-on-a-string coil into solenoid fibres.

Chains from Process 4's actual output (the nucleosome beads). The biology: the "beads on a string" (a
10 nm strand) coils into a **~30 nm fibre** at **~6 nucleosomes per turn** — this is where the packing
finally starts to *funnel*. Grouping one row of beads per fibre (~36 beads → ~6 turns), the flat carpet
of 1663 beads becomes a field of **~47 thirty-nm fibres** — the first real drop in count (1663 → 47) and
a ~6× further compaction, the DNA beginning to look like a chromosome arm.

Conserving and physical: each nucleosome bead moves as a **rigid unit** — its wrapped ring of DNA is
carried along, translated onto the solenoid — so every base pair is reused, nothing is spawned, nothing
teleports. This lib supplies the two end states (bead carpet → fibre); ``scenes/warp_fibre`` animates the
coil.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from .nucleosome import wrap_nucleosomes


@dataclasses.dataclass
class Fibre:
    """The base pairs in two states: ``bead_a`` / ``bead_b`` (P,3) as Process 4 left them (beads on a
    string), and ``fib_a`` / ``fib_b`` (P,3) coiled into 30 nm fibres. ``a_col`` / ``b_col`` base colours,
    ``centers`` (F,3) the fibre-bead centres after coiling, ``beads_per_turn`` the solenoid pitch,
    ``fibre_radius`` its radius, ``n_fibres`` how many fibres the beads funnelled into."""

    bead_a: np.ndarray
    bead_b: np.ndarray
    fib_a: np.ndarray
    fib_b: np.ndarray
    a_col: np.ndarray
    b_col: np.ndarray
    centers: np.ndarray
    beads_per_turn: float
    fibre_radius: float
    n_fibres: int
    beads_per_fibre: int = 0
    bp_per_bead: int = 0

    @property
    def n_pairs(self) -> int:
        return int(self.bead_a.shape[0])


def coil_fibre(sub: int = 2, block: int = 5, beads_per_turn: float = 6.0,
               fibre_radius: float = 1.05, axial_pitch: float = 0.185,
               fibre_spacing: float = 3.5, beads_per_fibre: int = 72) -> Fibre:
    """Coil Process 4's beads-on-a-string into 30 nm solenoid fibres — ~6 nucleosomes per turn, tightly
    coiled and compacted along the (vertical) fibre axis. Each bead is rigid-moved onto the solenoid; a
    long run of the string makes one long rope, so the 1663 beads **funnel into a couple dozen 30 nm
    fibres** — the first real drop in count and a strong compaction, the DNA beginning to look like a
    chromosome arm. The result is a forest of thick coiled ropes, paralleling Process 3's forest of thin
    helices: thin threads -> flat beads -> coiled ropes.

    **Physically sized so nothing interpenetrates** (verified with ``genome.strand.min_separation``):
    - the fibre radius makes the turn's circumference ``2*pi*R`` exceed ``beads_per_turn * bead_diameter``
      (~0.94), so the ~6 nucleosomes sit around each turn with linker gaps instead of overlapping —
      the real ~30 nm-fibre-to-11 nm-nucleosome ratio (~2.7×);
    - the axial pitch is the real tight solenoid rise (~one bead diameter per turn), so consecutive turns
      just clear each other (``beads_per_turn * axial_pitch`` >= a bead diameter) — a dense rope, not a
      stretched spring;
    - the fibres stand on a serpentine 2-D grid spaced ``fibre_spacing`` (> the fibre's outer diameter),
      so neighbouring ropes never touch, and the continuous strand snakes rope-to-rope (each rope winds
      the opposite way in height, so one rope's top meets the next rope's top — a short local link)."""
    nc = wrap_nucleosomes(sub=sub, block=block)
    centers = nc.centers                                   # (H,3) bead centres from Process 4
    h = nc.n_beads
    g = nc.bp_per_nuc
    k = int(beads_per_fibre)

    bead_id = np.arange(h)
    f = bead_id // k                                       # fibre index (a long run of beads → one rope)
    local = bead_id % k                                    # position of the bead along its fibre
    n_fibres = int(f.max()) + 1

    # boustrophedon in HEIGHT: even ropes wind bottom→top, odd ropes top→bottom, so the end of one rope
    # meets the start of the next at the same height (a short link), keeping one continuous strand.
    up = (f % 2) == 0
    jeff = np.where(up, local, k - 1.0 - local).astype(np.float32)
    phi = jeff * (2.0 * math.pi / beads_per_turn)          # ~6 beads per turn
    j0 = jeff - (k - 1.0) * 0.5                             # centre the coil on the vertical fibre axis

    # stand the ropes in a forest: a serpentine 2-D grid so consecutive ropes are neighbours, each cell
    # wider than a rope's outer diameter so ropes never touch.
    fx = max(int(round(math.sqrt(n_fibres))), 1)
    fi = np.arange(n_fibres)
    frow = fi // fx
    fcol = np.where(frow % 2 == 0, fi % fx, fx - 1 - (fi % fx))
    fz = (n_fibres + fx - 1) // fx
    gx = (fcol - (fx - 1) * 0.5) * fibre_spacing
    gz = (frow - (fz - 1) * 0.5) * fibre_spacing
    cx = gx[f] + fibre_radius * np.cos(phi)
    cy = j0 * axial_pitch                                  # rise along the vertical fibre axis
    cz = gz[f] + fibre_radius * np.sin(phi)
    new_center = np.stack([cx, cy, cz], 1).astype(np.float32)

    # each bead's wrapped ring rigid-moves onto the solenoid; its linker DNA is re-routed to the bead's
    # NEW neighbours (so nothing stretches across the band) — both conserving, every base pair reused.
    # the two free ends of the whole strand have no neighbour, so extrapolate one (a real short linker
    # sticking out) instead of collapsing the linker onto the centre.
    i = np.arange(nc.n_pairs)
    bead = i // g
    s = (i % g).astype(np.float32) / float(g)
    prev_id = bead - 1
    next_id = bead + 1
    prev_c = new_center[np.clip(prev_id, 0, h - 1)]
    next_c = new_center[np.clip(next_id, 0, h - 1)]
    prev_c[prev_id < 0] = (2.0 * new_center[0] - new_center[1])          # extrapolate the leading free end
    next_c[next_id > h - 1] = (2.0 * new_center[h - 1] - new_center[h - 2])   # trailing free end
    shift = (new_center - centers)[bead]
    fib_a = nc.nuc_a + shift
    fib_b = nc.nuc_b + shift

    lo, hi = nc.link_frac, 1.0 - nc.link_frac
    m_in = s < lo
    t_in = np.clip(s / max(lo, 1e-6), 0.0, 1.0)[:, None]
    lin_in = 0.5 * (new_center[bead] + prev_c) * (1.0 - t_in) + new_center[bead] * t_in
    m_out = s > hi
    t_out = np.clip((s - hi) / max(1.0 - hi, 1e-6), 0.0, 1.0)[:, None]
    lin_out = new_center[bead] * (1.0 - t_out) + 0.5 * (new_center[bead] + next_c) * t_out
    fib_a[m_in] = lin_in[m_in]; fib_b[m_in] = lin_in[m_in]
    fib_a[m_out] = lin_out[m_out]; fib_b[m_out] = lin_out[m_out]

    return Fibre(bead_a=nc.nuc_a, bead_b=nc.nuc_b, fib_a=fib_a.astype(np.float32),
                 fib_b=fib_b.astype(np.float32), a_col=nc.a_col, b_col=nc.b_col, centers=new_center,
                 beads_per_turn=beads_per_turn, fibre_radius=fibre_radius, n_fibres=n_fibres,
                 beads_per_fibre=k, bp_per_bead=g)
