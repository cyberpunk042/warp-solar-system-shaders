"""N-body gravity — a real O(N²) gravitational simulation on Warp.

Every particle pulls on every other by Newton's law, softened at short range so the
integration stays stable through close passes. Warp evaluates the whole force matrix in
parallel each step; a leapfrog (kick-drift) integrator advances the state. Set up as two
gravitationally bound clumps on a collision course — they fall together, tear tidal tails,
and merge into one relaxed cluster with a dense core. Used by ``scenes/nbody.py``.
"""

import numpy as np
import warp as wp


@wp.kernel
def _accel(pos: wp.array(dtype=wp.vec3), mass: wp.array(dtype=float),
           acc: wp.array(dtype=wp.vec3), g: float, eps2: float, n: int):
    i = wp.tid()
    pi = pos[i]
    a = wp.vec3(0.0, 0.0, 0.0)
    for j in range(n):
        d = pos[j] - pi
        r2 = wp.dot(d, d) + eps2
        inv = 1.0 / (r2 * wp.sqrt(r2))          # 1 / (r²+ε²)^{3/2}
        a = a + d * (g * mass[j] * inv)
    acc[i] = a


@wp.kernel
def _kick_drift(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3),
                acc: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid()
    v = vel[i] + acc[i] * dt
    vel[i] = v
    pos[i] = pos[i] + v * dt


def _plummer(n, radius, rng):
    """A Plummer-sphere-like cloud (dense core, falling density) of unit total mass."""
    # sample radius from a softened power law, random direction
    u = rng.random(n)
    r = radius / np.sqrt(np.maximum(u ** (-2.0 / 3.0) - 1.0, 1e-6))
    r = np.minimum(r, radius * 6.0)
    dirs = rng.normal(size=(n, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9
    return (dirs * r[:, None]).astype(np.float32)


def make_collision(n=4000, sep=2.6, radius=0.6, approach=0.32, spin=0.5, impact=0.0, seed=7):
    """Two Plummer clumps offset on x (and by ``impact`` on y for a grazing pass),
    drifting toward each other, each spinning — a setup that throws graceful tidal tails."""
    rng = np.random.default_rng(seed)
    n2 = n // 2
    pa = _plummer(n2, radius, rng)
    pb = _plummer(n - n2, radius, rng)
    pa[:, 0] -= sep * 0.5
    pb[:, 0] += sep * 0.5
    pa[:, 1] += impact * 0.5
    pb[:, 1] -= impact * 0.5
    # internal rotation about z (gives tidal tails a graceful curl)
    va = np.zeros_like(pa)
    vb = np.zeros_like(pb)
    va[:, 0] = -pa[:, 1] * spin
    va[:, 1] = (pa[:, 0] + sep * 0.5) * spin
    vb[:, 0] = -pb[:, 1] * spin
    vb[:, 1] = (pb[:, 0] - sep * 0.5) * spin
    # bulk approach velocity along x
    va[:, 0] += approach
    vb[:, 0] -= approach
    pos = np.concatenate([pa, pb]).astype(np.float32)
    vel = np.concatenate([va, vb]).astype(np.float32)
    clump = np.concatenate([np.zeros(n2, np.int32), np.ones(n - n2, np.int32)])
    mass = np.full(n, 1.0 / float(n), np.float32)
    return pos, vel, mass, clump


class NBody:
    def __init__(self, pos, vel, mass, device="cpu", g=1.0, eps=0.06):
        self.n = len(pos)
        self.device = device
        self.g = float(g)
        self.eps2 = float(eps * eps)
        self.pos = pos.copy()
        self.vel = vel.copy()
        self._d_pos = wp.array(pos, dtype=wp.vec3, device=device)
        self._d_vel = wp.array(vel, dtype=wp.vec3, device=device)
        self._d_mass = wp.array(mass, dtype=float, device=device)
        self._d_acc = wp.zeros(self.n, dtype=wp.vec3, device=device)

    def step(self, dt):
        wp.launch(_accel, dim=self.n,
                  inputs=[self._d_pos, self._d_mass, self._d_acc, self.g, self.eps2, self.n],
                  device=self.device)
        wp.launch(_kick_drift, dim=self.n,
                  inputs=[self._d_pos, self._d_vel, self._d_acc, float(dt)],
                  device=self.device)

    def run(self, steps, dt):
        for _ in range(steps):
            self.step(dt)
        wp.synchronize_device(self.device)
        self.pos = self._d_pos.numpy()
        self.vel = self._d_vel.numpy()
        return self.pos, self.vel
