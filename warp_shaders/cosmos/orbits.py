"""Orbital mechanics — Keplerian ellipses for the stable case, N-body gravity
for the destructive one, and the merger / collapse physics that decides which
celestial body a growing mass becomes.

Two regimes, by design:

* **stable** (default) — each body follows a fixed Keplerian ellipse
  (`orbit_position`), solving Kepler's equation each frame. No integration
  drift; a system laid out this way orbits forever, cleanly.
* **destructive** — bodies are integrated as point masses under mutual gravity
  (`nbody_step`, velocity-Verlet). Bodies that touch **merge** (momentum-
  conserving); a black hole **swallows** anything inside its horizon. A merged
  mass becomes its correct remnant by mass (`remnant_type`): a bigger/hotter
  star, then — past the collapse thresholds — a neutron star, then a black hole,
  with a supernova flash on the collapse transition.

All positions use the scene's Y-up convention; an orbit with ``incl=0`` lies in
the XZ plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# body masses in solar masses (also the visual mass scale)
MASS = {"sun": 1.0, "neutron": 1.6, "white_dwarf": 0.6, "black_hole": 8.0}

# collapse thresholds (solar masses) — Chandrasekhar / TOV, dramatised for view
M_BLUE = 8.0        # above this a merged star is a hot blue giant
M_NEUTRON = 25.0    # collapse to a neutron star (supernova)
M_BLACK_HOLE = 45.0  # collapse to a black hole


@dataclass
class Orbit:
    """Keplerian elements. ``period<=0`` pins the body at the focus (a central
    star). Distances are scene units; angles radians; period in time units."""
    a: float = 4.0        # semi-major axis
    e: float = 0.0        # eccentricity (0 circle .. <1 ellipse)
    incl: float = 0.0     # inclination
    node: float = 0.0     # longitude of ascending node
    arg: float = 0.0      # argument of periapsis
    period: float = 6.0   # orbital period
    phase: float = 0.0    # mean anomaly at t=0


def solve_kepler(M: float, e: float, iters: int = 8) -> float:
    """Eccentric anomaly E from mean anomaly M: M = E - e·sin E (Newton)."""
    M = (M + math.pi) % (2.0 * math.pi) - math.pi
    E = M if e < 0.8 else math.pi
    for _ in range(iters):
        f = E - e * math.sin(E) - M
        E = E - f / (1.0 - e * math.cos(E))
    return E


def orbit_position(orb: Orbit, time: float) -> np.ndarray:
    """World position of a body on orbit `orb` at `time` (Y-up, XZ plane at
    incl=0)."""
    if orb.period <= 0.0:
        return np.zeros(3, np.float32)
    M = orb.phase + 2.0 * math.pi * time / orb.period
    E = solve_kepler(M, orb.e)
    x = orb.a * (math.cos(E) - orb.e)
    z = orb.a * math.sqrt(max(1.0 - orb.e * orb.e, 0.0)) * math.sin(E)
    # argument of periapsis (rotate in the orbital plane, about Y)
    ca, sa = math.cos(orb.arg), math.sin(orb.arg)
    xp = x * ca - z * sa
    zp = x * sa + z * ca
    # inclination (tilt about X, lifting into Y)
    ci, si = math.cos(orb.incl), math.sin(orb.incl)
    xi, yi, zi = xp, zp * si, zp * ci
    # longitude of ascending node (rotate about Y)
    cn, sn = math.cos(orb.node), math.sin(orb.node)
    X = xi * cn + zi * sn
    Z = -xi * sn + zi * cn
    return np.array([X, yi, Z], np.float32)


def orbit_velocity(orb: Orbit, time: float, dt: float = 1e-3) -> np.ndarray:
    """Finite-difference velocity along the orbit (for seeding an N-body run)."""
    p0 = orbit_position(orb, time - dt)
    p1 = orbit_position(orb, time + dt)
    return (p1 - p0) / (2.0 * dt)


def circular_speed(m_central: float, r: float, g: float = 1.0) -> float:
    """Speed for a circular orbit of radius `r` about mass `m_central`."""
    return math.sqrt(g * m_central / max(r, 1e-4))


# --------------------------------------------------------------------------- #
# N-body (destructive regime)                                                 #
# --------------------------------------------------------------------------- #

def nbody_accel(pos: np.ndarray, mass: np.ndarray, g: float = 1.0,
                soft: float = 0.15) -> np.ndarray:
    n = len(mass)
    acc = np.zeros((n, 3), np.float32)
    for i in range(n):
        d = pos - pos[i]                       # (n, 3)
        r2 = (d * d).sum(1) + soft * soft
        r2[i] = 1.0e18
        inv = mass / (r2 * np.sqrt(r2))
        acc[i] = g * (d * inv[:, None]).sum(0)
    return acc


def nbody_step(pos, vel, mass, dt, g=1.0, soft=0.15):
    """One velocity-Verlet (KDK) step of mutual gravity."""
    a0 = nbody_accel(pos, mass, g, soft)
    vel = vel + 0.5 * dt * a0
    pos = pos + dt * vel
    a1 = nbody_accel(pos, mass, g, soft)
    vel = vel + 0.5 * dt * a1
    return pos.astype(np.float32), vel.astype(np.float32)


def remnant_type(mass: float) -> str:
    """The celestial body a (merged) mass becomes."""
    if mass >= M_BLACK_HOLE:
        return "black_hole"
    if mass >= M_NEUTRON:
        return "neutron"
    return "sun"                               # main-sequence (blue if massive)


def is_collapse(mass_before: float, mass_after: float) -> str:
    """Did crossing from `mass_before` to `mass_after` trigger a collapse? Returns
    '' / 'neutron' / 'black_hole' (the new remnant kind), for the flash event."""
    b, a = remnant_type(mass_before), remnant_type(mass_after)
    if b != a and a in ("neutron", "black_hole"):
        return a
    return ""
