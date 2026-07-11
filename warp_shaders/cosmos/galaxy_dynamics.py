"""Colliding galaxies — a Toomre restricted N-body encounter (host, NumPy).

Two point-mass **cores** attract each other with softened gravity and move on a
fly-by orbit; each is ringed by a disk of **massless test particles** on initially
circular orbits that feel *both* cores but exert no force. Prograde disks throw
long tidal **tails** and a **bridge**; retrograde disks barely respond. See
``docs/research/12-galaxy-collisions.md``.

`simulate(enc, frames, ...)` integrates the whole fly-by with velocity-Verlet and
returns a :class:`Collision` — per-frame particle + core positions, plus each
particle's galaxy id and colour — ready for the splat renderer.
"""

from __future__ import annotations

import dataclasses

import numpy as np

_G = 1.0


@dataclasses.dataclass
class GalaxyConfig:
    mass: float = 1.0
    n: int = 1800                       # test particles in the disk
    r_in: float = 0.4
    r_out: float = 2.2
    incl_deg: float = 20.0              # disk tilt to the orbit (xy) plane
    spin: float = 1.0                   # +1 prograde, -1 retrograde
    center: tuple = (0.0, 0.0, 0.0)     # initial core position
    vel: tuple = (0.0, 0.0, 0.0)        # initial core velocity
    color: tuple = (1.0, 0.85, 0.6)     # disk star colour


@dataclasses.dataclass
class EncounterConfig:
    g0: GalaxyConfig
    g1: GalaxyConfig
    soft: float = 0.25                  # gravitational softening length
    seed: int = 3


@dataclasses.dataclass
class Collision:
    part_pos: np.ndarray                # (frames, N, 3)
    core_pos: np.ndarray                # (frames, 2, 3)
    gal_id: np.ndarray                  # (N,) 0 or 1
    color: np.ndarray                   # (N, 3)
    core_mass: np.ndarray               # (2,)

    @property
    def frames(self):
        return self.part_pos.shape[0]


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def _make_disk(gc: GalaxyConfig, rng):
    n = gc.n
    # sqrt-distributed radii -> roughly uniform surface density in the disk
    r = np.sqrt(rng.uniform(gc.r_in ** 2, gc.r_out ** 2, n))
    th = rng.uniform(0.0, 2.0 * np.pi, n)
    pos = np.stack([r * np.cos(th), r * np.sin(th), np.zeros(n)], axis=1)
    vc = np.sqrt(_G * gc.mass / r)                      # circular speed
    vel = np.stack([-np.sin(th), np.cos(th), np.zeros(n)], axis=1) * (vc * gc.spin)[:, None]
    R = _rot_x(gc.incl_deg)
    pos = pos @ R.T
    vel = vel @ R.T
    pos += np.asarray(gc.center, np.float64)
    vel += np.asarray(gc.vel, np.float64)
    return pos, vel


def _accel(pos, cores, cmass, soft2):
    """Softened acceleration on `pos` (M,3) from the two `cores` (2,3)."""
    d = cores[None, :, :] - pos[:, None, :]            # (M, 2, 3)
    r2 = (d * d).sum(-1) + soft2                        # (M, 2)
    inv = (_G * cmass)[None, :] * r2 ** -1.5            # (M, 2)
    return (d * inv[..., None]).sum(1)                  # (M, 3)


def _core_accel(cpos, cmass, soft2):
    a = np.zeros((2, 3))
    d = cpos[1] - cpos[0]
    r3 = (d @ d + soft2) ** 1.5
    a[0] = _G * cmass[1] * d / r3
    a[1] = -_G * cmass[0] * d / r3
    return a


def simulate(enc: EncounterConfig, frames: int = 60, substeps: int = 10,
             dt: float = 0.06) -> Collision:
    """Integrate the encounter; record `frames` snapshots (`substeps` Verlet steps
    between each). Returns a :class:`Collision`."""
    rng = np.random.default_rng(enc.seed)
    p0, v0 = _make_disk(enc.g0, rng)
    p1, v1 = _make_disk(enc.g1, rng)
    part = np.concatenate([p0, p1], 0)
    pvel = np.concatenate([v0, v1], 0)
    gal_id = np.concatenate([np.zeros(enc.g0.n, int), np.ones(enc.g1.n, int)])
    color = np.concatenate([np.tile(enc.g0.color, (enc.g0.n, 1)),
                            np.tile(enc.g1.color, (enc.g1.n, 1))]).astype(np.float32)

    cpos = np.array([enc.g0.center, enc.g1.center], np.float64)
    cvel = np.array([enc.g0.vel, enc.g1.vel], np.float64)
    cmass = np.array([enc.g0.mass, enc.g1.mass], np.float64)
    soft2 = enc.soft ** 2

    pa = _accel(part, cpos, cmass, soft2)
    ca = _core_accel(cpos, cmass, soft2)

    P = np.zeros((frames, part.shape[0], 3), np.float32)
    C = np.zeros((frames, 2, 3), np.float32)
    for f in range(frames):
        P[f] = part
        C[f] = cpos
        for _ in range(substeps):
            # velocity-Verlet (KDK) for cores + particles in the cores' field
            cvel += 0.5 * dt * ca
            pvel += 0.5 * dt * pa
            cpos += dt * cvel
            part += dt * pvel
            ca = _core_accel(cpos, cmass, soft2)
            pa = _accel(part, cpos, cmass, soft2)
            cvel += 0.5 * dt * ca
            pvel += 0.5 * dt * pa
    return Collision(P, C, gal_id, color, cmass.astype(np.float32))
