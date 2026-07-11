"""Superposition → collapse — the wave/collapse metaphor over plant futures.

The operator's framing: *"how things are waves before what to us seems like a
collapse in the world … the mind will have even more impact 'backward' in time,
in the snapshot / visual part of time like 1/3000000."* This module realises that
as an **ensemble of possible plant futures** (several stochastic / different
grammars, all rooted at the same spot) that begin **superposed** — faint,
overlapping ghosts, a cloud of what the plant *might* become — and then
**collapse** to a single realised plant. A "mind" biases *which* future resolves.

This is an explicit **metaphor**, not literal quantum mechanics: the ensemble is
the wavefunction, the mean image is the superposition, and picking one member is
the measurement. The collapse sweeps a **front** through the frame so the plant
crystallises tip→base — the future settling first and reaching *backward* into
its own history.

Pure NumPy (no Warp) so the blend is unit-testable; the scene supplies the
rendered ensemble images.
"""

from __future__ import annotations

import numpy as np


def superpose(images, weights=None) -> np.ndarray:
    """Weighted superposition (mean) of an ensemble of ``(H, W, 3)`` images.

    With no weights this is the uniform mean — the faint overlapping ghost cloud.
    """
    imgs = np.stack(images, 0).astype(np.float32)
    if weights is None:
        return imgs.mean(0)
    w = np.asarray(weights, np.float32)
    w = w / (w.sum() + 1e-9)
    return np.tensordot(w, imgs, axes=(0, 0)).astype(np.float32)


def collapse_blend(ghosts, chosen_idx: int, front_frac: float,
                   band: float = 0.14, row_range=None) -> np.ndarray:
    """Blend a superposed ensemble into one realised plant along a sweeping front.

    Rows above the collapse front show the chosen realised plant; rows below stay
    superposed (the ghost cloud); `band` is the soft transition width. `front_frac`
    0→1 sweeps the front from the top of `row_range` to its bottom, so the plant
    crystallises from its tips (the future) down to its base (the past). Confine
    `row_range=(r0, r1)` to the subject's pixel extent so the collapse is spent on
    the plant, not on empty sky above it (defaults to the whole image).
    """
    ghosts = [g.astype(np.float32) for g in ghosts]
    h, w, _ = ghosts[0].shape
    cloud = superpose(ghosts)
    chosen = ghosts[int(chosen_idx) % len(ghosts)]
    r0, r1 = (0.0, float(h)) if row_range is None else row_range
    span = r1 - r0
    denom = band * span + 1e-6
    rows = np.arange(h, dtype=np.float32)
    # sweep the front from r0 to just past r1 (by one band) so front_frac=1
    # fully collapses the last rows instead of leaving them mid-transition
    front = r0 + front_frac * (span + denom)
    # alpha: 1 well above the front (collapsed), 0 below (still cloud)
    x = (front - rows) / denom
    x = np.clip(x, 0.0, 1.0)
    alpha = (x * x * (3.0 - 2.0 * x))[:, None, None]      # smoothstep per row
    return (alpha * chosen + (1.0 - alpha) * cloud).astype(np.float32)


def pick_index(drives) -> int:
    """The future the mind favours: the band with the strongest drive."""
    return int(np.argmax(np.asarray(drives, np.float32)))
