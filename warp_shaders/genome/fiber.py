"""Process 5 — the 30nm fiber: coil the beads-on-a-string into a solenoid.

A separate conserving process. Its INPUT is the nucleosome beads-on-a-string from Process 4. The string
is still too long, so it does the next level of packing: the string itself **coils into a solenoid** —
~6 nucleosomes per turn wound around a common axis into the thick **30nm chromatin fibre**.

Conserving and physical: not one base pair is created or destroyed — each nucleosome bead is carried,
rigidly and whole, from its place on the string onto the solenoid (a translation of the bead; its inner
wrap is untouched). Every base pair placed exactly once; the string reels continuously onto the fibre.
This process stops at the 30nm fibre — folding it into the chromosome is the next process.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .nucleosome import wrap_nucleosomes, _PER, _WRAP

_PER_TURN = 6.0      # nucleosomes per turn of the solenoid
_RFIB = 1.9          # fibre radius (beads wound around the axis)
_ADV = 0.52          # axial advance per bead


@dataclasses.dataclass
class Fiber:
    """The beads-on-a-string coiled into the 30nm solenoid. ``string`` (P,3) is the beads-on-a-string
    input; ``fiber`` (P,3) the same base pairs with each bead carried onto the solenoid; ``colors``
    (P,3). Conserved: P points for P base pairs, each bead translated whole — none spawned."""

    string: np.ndarray
    fiber: np.ndarray
    colors: np.ndarray
    n_beads: int

    @property
    def n_pairs(self) -> int:
        return int(self.string.shape[0])


def coil_fiber(sub: int = 2, block: int = 5) -> Fiber:
    """Coil the Process-4 nucleosome string into the 30nm solenoid fibre. Returns :class:`Fiber`."""
    nc = wrap_nucleosomes(sub=sub, block=block)
    string = nc.wrapped                      # beads-on-a-string (Process-4 output)
    centers = nc.centers                     # (n_beads,3) string bead centres
    bi = nc.bead_index                       # (P,) bead per base pair
    nb = nc.n_beads

    # solenoid bead centres: ~6 nucleosomes per turn wound around the x-axis
    bn = np.arange(nb)
    ang = bn * (2.0 * np.pi / _PER_TURN)
    ax = bn * _ADV
    ax = ax - ax.mean()
    fib_centers = np.stack([ax, _RFIB * np.cos(ang), _RFIB * np.sin(ang)], axis=1).astype(np.float32)

    # carry each bead from its string centre to its solenoid centre. Wrapped points move with their own
    # bead; a linker point blends from its bead's shift to the NEXT bead's shift across the linker, so it
    # stretches correctly between the two relocated beads (no dangling connectors).
    bead_shift = (fib_centers - centers)                     # (n_beads,3)
    i = np.arange(string.shape[0])
    l = i % _PER
    f = np.clip((l - _WRAP) / float(_PER - _WRAP), 0.0, 1.0)  # 0 across the wrap, 0->1 across the linker
    n1 = np.clip(bi + 1, 0, nb - 1)
    shift = bead_shift[bi] * (1.0 - f)[:, None] + bead_shift[n1] * f[:, None]
    fiber = (string + shift).astype(np.float32)

    return Fiber(string=string, fiber=fiber, colors=nc.colors, n_beads=nb)
