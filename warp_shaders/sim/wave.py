"""The wave equation — a 2-D finite-difference solver on a grid.

Solves ``u_tt = c² ∇²u`` (with a little damping) by the standard explicit leapfrog stencil:
the next height at each cell comes from the current and previous heights plus the discrete
Laplacian, `u_next = 2u − u_prev + (c·dt/h)² ∇²u`. Point **oscillators** drive the field at
fixed cells, so two of them in phase produce the textbook **two-source interference** pattern —
hyperbolic nodal lines where crest meets trough. A tapered damping border absorbs the outgoing
waves so they don't reflect off the edges. Used by ``scenes/ripple_tank.py``.
"""

import numpy as np


class WaveField:
    def __init__(self, n=240, c=0.5, damp=0.9985, border=0.10):
        self.n = n
        self.c2 = float(c) * float(c)            # Courant factor (dt=h=1); keep c ≤ ~0.7 for stability
        self.damp = float(damp)
        self.u = np.zeros((n, n), np.float32)
        self.u_prev = np.zeros((n, n), np.float32)
        self.sources = []                        # (i, j, amp, omega, phase)
        self.wall = None                         # optional boolean barrier mask (u pinned to 0)
        # absorbing border: multiplier ramps from 1 (interior) to <1 (edge)
        yy, xx = np.mgrid[0:n, 0:n]
        d = np.minimum.reduce([xx, yy, n - 1 - xx, n - 1 - yy]).astype(np.float32) / (border * n)
        self.absorb = (0.86 + 0.14 * np.clip(d, 0.0, 1.0)).astype(np.float32)

    def add_source(self, x, y, amp=1.0, omega=0.32, phase=0.0):
        self.sources.append((int(y * self.n), int(x * self.n), float(amp), float(omega), float(phase)))

    def add_line_source(self, y, amp=1.0, omega=0.32, x0=0.05, x1=0.95, phase=0.0):
        """A row of in-phase oscillators — a plane-wave emitter."""
        i = int(y * self.n)
        for jx in range(int(x0 * self.n), int(x1 * self.n)):
            self.sources.append((i, jx, float(amp), float(omega), float(phase)))

    def double_slit(self, y, gap=0.05, sep=0.16, thickness=0.02):
        """A barrier row at height ``y`` with two slit openings (a reflecting wall)."""
        n = self.n
        wall = np.zeros((n, n), bool)
        i0 = int((y - thickness * 0.5) * n)
        i1 = int((y + thickness * 0.5) * n) + 1
        wall[i0:i1, :] = True
        for cx in (0.5 - sep * 0.5, 0.5 + sep * 0.5):
            j0 = int((cx - gap * 0.5) * n)
            j1 = int((cx + gap * 0.5) * n)
            wall[i0:i1, j0:j1] = False
        self.wall = wall

    def step(self, t):
        lap = (np.roll(self.u, 1, 0) + np.roll(self.u, -1, 0)
               + np.roll(self.u, 1, 1) + np.roll(self.u, -1, 1) - 4.0 * self.u)
        u_next = (2.0 * self.u - self.u_prev + self.c2 * lap) * self.damp
        u_next *= self.absorb
        if self.wall is not None:
            u_next[self.wall] = 0.0
        for i, j, amp, omega, phase in self.sources:
            u_next[i, j] = amp * np.sin(omega * t + phase)
        self.u_prev = self.u
        self.u = u_next

    def run(self, steps):
        for k in range(steps):
            self.step(float(k))
        return self.u

    def laplacian(self):
        return (np.roll(self.u, 1, 0) + np.roll(self.u, -1, 0)
                + np.roll(self.u, 1, 1) + np.roll(self.u, -1, 1) - 4.0 * self.u)
