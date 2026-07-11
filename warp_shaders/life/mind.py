"""A tiny mind — Conway's Game of Life as a decision substrate.

Before a mind, the plant obeys the "obvious rules" reflexively (it *always*
bends to the light, *always* closes in the rain). A mind lets it **choose**. The
minimal computational substrate for "choice from local rules" is a cellular
automaton: Conway's Game of Life (Gardner 1970) is Turing-complete, so an
arbitrarily complex deliberation can in principle run on this grid.

`Mind` runs a toroidal Life grid and exposes a bounded **drive** in [0, 1] read
off the living population — the aggregate "activity" of the little mind. The
plant scene maps that drive onto the tropism knobs: high drive ⇒ *seek the
light* (open, phototropic); low drive ⇒ *rest* (sag, leaves folded). The CA is
kept alive by a periodic stimulus (a sensory poke), so the drive waxes and wanes
and the plant visibly switches between chasing the light and closing up.

Deterministic from ``seed`` — the same seed replays the same deliberation.
"""

from __future__ import annotations

import numpy as np


def _smoothstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


class Mind:
    def __init__(self, size: int = 40, seed: int = 0, density: float = 0.30,
                 stimulus_every: int = 6, lo: float = 0.07, hi: float = 0.17):
        self.size = size
        self._rng = np.random.default_rng(seed)
        self.grid = (self._rng.random((size, size)) < density).astype(np.uint8)
        self.gen = 0
        self.stimulus_every = stimulus_every
        self.lo, self.hi = lo, hi

    def step(self) -> None:
        """One Conway generation (B3/S23) on a toroidal grid, + periodic poke."""
        g = self.grid
        nb = sum(np.roll(np.roll(g, dy, 0), dx, 1)
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if not (dx == 0 and dy == 0))
        self.grid = ((nb == 3) | ((g == 1) & (nb == 2))).astype(np.uint8)
        self.gen += 1
        if self.stimulus_every and self.gen % self.stimulus_every == 0:
            # a sensory poke keeps the deliberation from dying out: seed a small
            # random burst in a random cell of the grid (deterministic per seed)
            s = self.size
            r0 = int(self._rng.integers(0, s - 5))
            c0 = int(self._rng.integers(0, s - 5))
            burst = (self._rng.random((5, 5)) < 0.5).astype(np.uint8)
            self.grid[r0:r0 + 5, c0:c0 + 5] |= burst

    def population(self) -> float:
        """Fraction of live cells."""
        return float(self.grid.mean())

    def decision(self) -> float:
        """Bounded drive in [0, 1]: smoothstep of live-fraction over [lo, hi].

        High drive = active mind = *seek the light*; low = quiescent = *rest*.
        """
        return _smoothstep((self.population() - self.lo) / (self.hi - self.lo))

    def decisions(self, k: int) -> list:
        """`k` independent drives, one per vertical band of the grid — so
        different *parts* of one plant can be steered by different regions of the
        same mind (a per-branch decision, not a whole-plant one)."""
        bands = np.array_split(self.grid, k, axis=1)
        return [_smoothstep((float(b.mean()) - self.lo) / (self.hi - self.lo))
                for b in bands]


def run_to(size: int, seed: int, steps: int) -> Mind:
    """Fresh `Mind` advanced `steps` generations (scenes are stateless per frame)."""
    m = Mind(size=size, seed=seed)
    for _ in range(max(steps, 0)):
        m.step()
    return m
