"""Process 6 — the metaphase chromosome: fold the fibre into the blue X.

The last conserving process. Its INPUT is the 30nm fibre from Process 5. The fibre condenses and folds
into looped domains that pack the shape of the **metaphase chromosome**: the blue **X** — two chromatid
arms fat and rounded at the telomere tips, pinched at the **centromere** with its two lighter nodes.

Conserving and physical: every base pair (and so every token, every bit of the original card) is packed
into the chromosome body exactly once — nothing spawned, nothing dropped. Each point travels in a
straight, continuous line from its place on the fibre to its place in the X (no teleport). This is where
the whole ladder lands: the card, become a chromosome.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .fiber import coil_fiber

_S = 1.45                       # overall scale of the X
_CTX, _CTY = 0.56, 1.44         # arm tip direction (chromatid spread), pre-scale
_RMAX = 0.62                     # fattest arm radius
_ZFLAT = 0.6                     # cross-section depth (out-of-plane) vs width — a flattish chromosome
_CBODY = np.array([0.16, 0.47, 0.74], np.float32)    # chromosome blue
_CNODE = np.array([0.62, 0.82, 0.94], np.float32)    # lighter centromere nodes


@dataclasses.dataclass
class Chromosome:
    """The fibre folded into the metaphase X. ``fiber`` (P,3) is the 30nm-fibre input; ``chromo`` (P,3)
    the same base pairs packed into the chromosome body; ``colors`` (P,3) chromosome blue (lighter at
    the centromere nodes). Conserved: P points for P base pairs — none spawned."""

    fiber: np.ndarray
    chromo: np.ndarray
    colors: np.ndarray

    @property
    def n_pairs(self) -> int:
        return int(self.fiber.shape[0])


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def fold_chromosome(sub: int = 2, block: int = 5) -> Chromosome:
    """Fold the Process-5 fibre into the metaphase chromosome X. Returns :class:`Chromosome`."""
    fb = coil_fiber(sub=sub, block=block)
    fiber = fb.fiber
    p = fiber.shape[0]
    i = np.arange(p)

    # four chromatid arms (TL, TR, BL, BR) meeting at the centromere (origin)
    tips = np.array([[-_CTX, _CTY], [_CTX, _CTY], [-_CTX, -_CTY], [_CTX, -_CTY]], np.float64) * _S
    arm = (i * 4) // p                                   # which arm (contiguous quarters of the strand)
    u = (i * 4.0 / p) - arm                              # 0 at centromere -> 1 at the tip

    tip = np.zeros((p, 3))
    tip[:, :2] = tips[arm]
    axis_dir = _unit(tip)                                # centromere->tip direction (z=0)
    # in-plane perpendicular (arm width) and out-of-plane (thickness)
    p1 = _unit(np.cross(axis_dir, np.array([0.0, 0.0, 1.0])))
    p2 = np.array([0.0, 0.0, 1.0])

    # arm centreline: a gently bowed sweep from centromere to tip
    bow = 0.18 * _S * np.sin(u * np.pi)[:, None] * p1    # slight outward curve of each chromatid
    centre = u[:, None] * tip + bow

    # tapered radius: pinched at the centromere (u=0), fat mid, rounded at the tip
    rad = _RMAX * (0.28 + 0.72 * np.sin(np.pi * np.clip(0.08 + 0.86 * u, 0.0, 1.0)))

    # fill the tapered tube: a high-frequency spiral (phi) sweeping a radial fraction (rho, sqrt for a
    # uniform disc) — the fibre's looped domains packing the arm cross-section
    phi = i * 0.7003
    rho = np.sqrt((i * 0.113) % 1.0)
    off = (rad * rho)[:, None] * (np.cos(phi)[:, None] * p1 + _ZFLAT * np.sin(phi)[:, None] * p2)
    chromo = (centre + off).astype(np.float32)

    # colour: chromosome blue, lighter at the centromere nodes (near the pinch)
    node = np.clip(1.0 - u / 0.14, 0.0, 1.0) * (rho < 0.6)   # bright core near the centromere
    colors = (_CBODY[None, :] * (0.82 + 0.30 * rho[:, None])
              + (_CNODE - _CBODY)[None, :] * node[:, None]).astype(np.float32)
    colors = np.clip(colors, 0.0, 1.0)

    return Chromosome(fiber=fiber, chromo=chromo, colors=colors)
