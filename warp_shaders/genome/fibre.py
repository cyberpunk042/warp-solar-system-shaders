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
               fibre_radius: float = 0.72, axial_pitch: float = 0.30) -> Fibre:
    """Coil Process 4's beads-on-a-string into 30 nm solenoid fibres — one row of beads per fibre, ~6
    nucleosomes per turn, compacted along the fibre axis. Each bead is rigid-moved onto the solenoid."""
    nc = wrap_nucleosomes(sub=sub, block=block)
    nx = int(nc.grid_nx)
    centers = nc.centers                                   # (H,3) bead centres from Process 4
    h = nc.n_beads
    g = nc.bp_per_nuc

    bead_id = np.arange(h)
    f = bead_id // nx                                      # fibre index (one row of beads → one fibre)
    col = bead_id % nx                                     # position of the bead along its fibre
    n_fibres = int(f.max()) + 1
    cnt = np.bincount(f, minlength=n_fibres).astype(np.float32)
    mz = np.bincount(f, weights=centers[:, 2], minlength=n_fibres) / np.maximum(cnt, 1)

    # boustrophedon: even fibres run one way, odd fibres the other, so the end of one fibre meets the
    # start of the next — the continuous 30 nm fibre snakes through the stack with short links.
    even = (f % 2) == 0
    j = np.where(even, col, cnt[f] - 1.0 - col)            # index along the fibre (serpentine)
    j0 = j - (cnt[f] - 1.0) * 0.5                          # centre the coil on the fibre axis
    axis_x = j0 * axial_pitch                              # all fibres share a centred x axis (a tidy band)
    phi = j * (2.0 * math.pi / beads_per_turn)             # ~6 beads per turn
    new_center = np.stack([axis_x,
                           fibre_radius * np.cos(phi),
                           mz[f] + fibre_radius * np.sin(phi)], 1).astype(np.float32)

    # each bead's wrapped ring rigid-moves onto the solenoid; its linker DNA is re-routed to the bead's
    # NEW neighbours (so nothing stretches across the band) — both conserving, every base pair reused.
    i = np.arange(nc.n_pairs)
    bead = i // g
    s = (i % g).astype(np.float32) / float(g)
    prev_c = new_center[np.clip(bead - 1, 0, h - 1)]
    next_c = new_center[np.clip(bead + 1, 0, h - 1)]
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
                 beads_per_fibre=nx, bp_per_bead=g)
