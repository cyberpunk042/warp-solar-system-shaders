"""Stable fluids — a 2-D incompressible Navier–Stokes solver (Jos Stam, 1999).

A velocity field pushes a dye/heat field around a grid. Each step: **buoyancy** lifts hot
fluid, **vorticity confinement** puts the curl back that numerical diffusion steals (so the
smoke billows instead of smearing), the velocity is made **divergence-free** by a Poisson
pressure projection (Jacobi iterations), and everything is moved by **semi-Lagrangian
advection** (trace each cell backward, sample where it came from — unconditionally stable).
A hot emitter at the base drives a rising, curling smoke column. Used by ``scenes/fluid.py``.
"""

import numpy as np


def _bilerp(field, fx, fy):
    n = field.shape[0]
    fx = np.clip(fx, 0.5, n - 1.5)
    fy = np.clip(fy, 0.5, n - 1.5)
    i0 = np.floor(fx).astype(np.int64)
    j0 = np.floor(fy).astype(np.int64)
    i1 = i0 + 1
    j1 = j0 + 1
    s1 = fx - i0
    s0 = 1.0 - s1
    t1 = fy - j0
    t0 = 1.0 - t1
    return (s0 * (t0 * field[j0, i0] + t1 * field[j1, i0])
            + s1 * (t0 * field[j0, i1] + t1 * field[j1, i1]))


class StableFluid:
    def __init__(self, n=176, buoy=2.6, vort=7.0, seed=3):
        self.n = n
        self.buoy = buoy
        self.vort = vort
        self.vx = np.zeros((n, n), np.float32)
        self.vy = np.zeros((n, n), np.float32)
        self.d = np.zeros((n, n), np.float32)      # dye / smoke density
        self.t = np.zeros((n, n), np.float32)      # temperature
        self._rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:n, 0:n]
        self._xx = xx.astype(np.float32)
        self._yy = yy.astype(np.float32)

    def _project(self, iters=36):
        n = self.n
        div = -0.5 * ((np.roll(self.vx, -1, 1) - np.roll(self.vx, 1, 1))
                      + (np.roll(self.vy, -1, 0) - np.roll(self.vy, 1, 0))) / n
        p = np.zeros_like(div)
        for _ in range(iters):
            p = (div + np.roll(p, 1, 0) + np.roll(p, -1, 0)
                 + np.roll(p, 1, 1) + np.roll(p, -1, 1)) * 0.25
        self.vx -= 0.5 * n * (np.roll(p, -1, 1) - np.roll(p, 1, 1))
        self.vy -= 0.5 * n * (np.roll(p, -1, 0) - np.roll(p, 1, 0))
        self._walls()

    def _walls(self):
        self.vx[:, 0] = 0.0
        self.vx[:, -1] = 0.0
        self.vy[0, :] = 0.0
        self.vy[-1, :] = 0.0

    def _advect(self, field, dt):
        n = self.n
        fx = self._xx - dt * n * self.vx
        fy = self._yy - dt * n * self.vy
        return _bilerp(field, fx, fy)

    def _vorticity(self, dt):
        w = (0.5 * (np.roll(self.vy, -1, 1) - np.roll(self.vy, 1, 1))
             - 0.5 * (np.roll(self.vx, -1, 0) - np.roll(self.vx, 1, 0)))
        aw = np.abs(w)
        gx = 0.5 * (np.roll(aw, -1, 1) - np.roll(aw, 1, 1))
        gy = 0.5 * (np.roll(aw, -1, 0) - np.roll(aw, 1, 0))
        mag = np.sqrt(gx * gx + gy * gy) + 1e-5
        nx = gx / mag
        ny = gy / mag
        self.vx += self.vort * dt * (ny * w)
        self.vy += self.vort * dt * (-nx * w)

    def emit(self, dt, phase=0.0):
        n = self.n
        cx = n * 0.5
        cy = n - n * 0.12
        r = n * 0.055
        m = ((self._xx - cx) ** 2 + (self._yy - cy) ** 2) < r * r
        self.d[m] = np.minimum(self.d[m] + 1.0 * dt * 12.0, 1.2)
        self.t[m] = np.minimum(self.t[m] + 1.0 * dt * 12.0, 1.2)
        # a gentle symmetric wobble at the source so the column sways/billows without drifting
        wob = 0.32 * np.sin(phase * 3.1) + 0.14 * np.sin(phase * 7.7)
        self.vx[m] += wob
        self.vy[m] -= 0.5                                      # push upward (toward row 0)

    def step(self, dt=0.1, phase=0.0):
        self.emit(dt, phase)
        # buoyancy: hot & light rises (toward row 0 => negative vy)
        self.vy -= (self.t * self.buoy) * dt
        self._vorticity(dt)
        self._project()
        self.vx = self._advect(self.vx, dt)
        self.vy = self._advect(self.vy, dt)
        self._walls()
        self._project()
        self.d = self._advect(self.d, dt) * 0.994
        self.t = self._advect(self.t, dt) * 0.94       # temperature cools fast → glow stays low

    def run(self, steps, dt=0.1):
        for k in range(steps):
            self.step(dt, phase=float(k) * dt)
        return self.d, self.t, self.vx, self.vy
