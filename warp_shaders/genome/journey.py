"""The genome journey — all six conserving processes, chained into one continuous morph.

One point per base pair (all 182872), carried through the six stages the genome library builds one at a
time:

  tokens (the card) -> base pairs -> double helix -> nucleosomes -> 30nm fibre -> chromosome.

The endpoints are the real library shapes (the token cloud from Process 1, the chromosome X from Process
6). The intermediate stages are drawn as **compact, readable** versions of each structure (a few turns of
helix, a row of beads, a short solenoid) so the morph shows each stage's character instead of a
thousand-turn streak — a stylised overview of the ladder, every point conserved.
"""

from __future__ import annotations

import numpy as np

from .basepair import bind_pairs
from .chromosome import fold_chromosome, _CBODY

STAGE_NAMES = ("tokens", "base pairs", "double helix", "nucleosomes", "30nm fibre", "chromosome")


def _normalise(p, target=3.1):
    """Recentre to the origin and scale uniformly so the 90th-percentile radius is ``target``."""
    c = 0.5 * (np.percentile(p, 2, axis=0) + np.percentile(p, 98, axis=0))
    q = p - c
    r = np.percentile(np.linalg.norm(q, axis=1), 90)
    return (q * (target / max(r, 1e-6))).astype(np.float32)


def genome_journey(sub: int = 2, block: int = 5):
    """Build the six morph keyframes (K, N, 3) + per-point colours + the chromosome blue."""
    bp = bind_pairs(sub=sub, block=block)
    n = bp.n_pairs
    i = np.arange(n)
    s = (i + 0.5) / n                                   # arc parameter along the strand, 0..1
    tau = 2.0 * np.pi

    # stage 0 — tokens: the real dispersed cloud (each base pair's two tokens' midpoint)
    tokens = 0.5 * (bp.a_pos + bp.b_pos)

    # stage 1 — base pairs: a tidy ordered lattice (order emerging from the cloud)
    nx = int(round(n ** (1.0 / 3.0))) + 1
    pairs = np.stack([i % nx, (i // nx) % nx, i // (nx * nx)], axis=1).astype(np.float64)

    # stage 2 — double helix: two intertwined backbone strands, ~16 readable turns
    th = s * 16.0 * tau
    strand = (i % 2) * np.pi
    helix = np.stack([1.35 * np.cos(th + strand), (s - 0.5) * 6.2, 1.35 * np.sin(th + strand)], axis=1)

    # stage 3 — nucleosomes: ~12 beads on a string (each bead a short coil about the string axis)
    nb = 12.0
    b = np.floor(s * nb)
    local = s * nb - b
    ca = local * 1.8 * tau
    nx_str = (b / (nb - 1.0) - 0.5) * 6.4               # bead position along the string (x)
    nuc = np.stack([nx_str + 0.10 * np.cos(ca), 0.62 * np.cos(ca), 0.62 * np.sin(ca)], axis=1)

    # stage 4 — 30nm fibre: the beads wound onto a short solenoid (~4 turns)
    fth = (b / nb) * 4.0 * tau
    fx = (b / (nb - 1.0) - 0.5) * 5.2
    fibre = np.stack([fx + 0.10 * np.cos(ca),
                      1.5 * np.cos(fth) + 0.42 * np.cos(ca),
                      1.5 * np.sin(fth) + 0.42 * np.sin(ca)], axis=1)

    # stage 5 — chromosome: the real folded X (Process 6)
    chromo = fold_chromosome(sub=sub, block=block).chromo

    colors = (0.5 * (bp.a_col + bp.b_col)).astype(np.float32)
    # the chromosome (the planar X) is normalised a touch smaller so its arms sit inside the frame
    frames = [_normalise(tokens), _normalise(pairs), _normalise(helix),
              _normalise(nuc), _normalise(fibre), _normalise(chromo, target=2.1)]
    return np.stack(frames, axis=0), colors, _CBODY.copy()
