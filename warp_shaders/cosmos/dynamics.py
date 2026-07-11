"""Destructive dynamics — the N-body driver behind the collapsing scenarios.

The stable regime lays bodies on fixed Kepler ellipses. The **destructive**
regime integrates them as point masses under mutual gravity (`orbits.nbody_step`)
so they can actually interact:

* two stars that touch **merge** — momentum-conserving — into their correct
  remnant by combined mass (`orbits.remnant_type`): a bigger, hotter star; then,
  past the collapse thresholds, a **neutron star**, then a **black hole**, each
  crossing announced by a **flash** (a supernova for a collapse);
* a **black hole swallows** any body inside ~its horizon, adding the mass and
  momentum to itself and growing.

`simulate` steps the bodies and renders each frame through the normal
`render_system` (feeding it the live body set + N-body positions), compositing
the flashes on top. The result is a frame list a scene can play back.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .bodies import BLACK_HOLE, NEUTRON, SUN, WHITE_DWARF, make_star
from .orbits import (Orbit, is_collapse, nbody_step, orbit_position,
                     orbit_velocity, remnant_type)
from .system import (Planet, Star, SystemConfig, _cam_basis, _project,
                     render_system)

_PLANET = -1


@dataclass
class Body:
    kind: int
    mass: float
    radius: float
    pos: np.ndarray
    vel: np.ndarray
    temp: float = 0.5
    activity: float = 0.6
    spin: float = 1.0
    precess: float = 0.3
    seed: float = 1.0
    pcfg: object = None            # PlanetConfig if this is a planet
    alive: bool = True

    @property
    def is_planet(self):
        return self.kind == _PLANET


def _star_mass(cfg) -> float:
    base = {SUN: 6.0, NEUTRON: 1.6, WHITE_DWARF: 0.9, BLACK_HOLE: 30.0}
    return base.get(cfg.kind, 6.0) * (cfg.radius * cfg.radius)


def _bh_radius(mass: float) -> float:
    return 0.5 + 0.03 * mass                       # Schwarzschild radius ∝ mass


def _remnant_look(kind_name: str, mass: float):
    """(kind, radius, temp, activity) for a merged mass's remnant."""
    if kind_name == "black_hole":
        return BLACK_HOLE, _bh_radius(mass), 0.5, 1.0
    if kind_name == "neutron":
        return NEUTRON, 0.45, 0.92, 0.9
    # a (possibly blue) main-sequence star — bigger + hotter with mass
    radius = 0.6 * math.sqrt(mass / 6.0)
    temp = float(np.clip(0.42 + 0.018 * mass, 0.42, 0.95))
    return SUN, radius, temp, 0.7


def init_bodies(sys: SystemConfig, decay: float = 1.0) -> list:
    """Seed bodies from the system's orbits at t=0; `decay`<1 slows them so they
    spiral inward (an inspiral toward merger)."""
    bodies = []
    for s in sys.stars:
        p = orbit_position(s.orbit, 0.0)
        v = orbit_velocity(s.orbit, 0.0) * decay
        bodies.append(Body(kind=s.cfg.kind, mass=_star_mass(s.cfg),
                           radius=s.cfg.radius, pos=p.astype(np.float32),
                           vel=v.astype(np.float32), temp=s.cfg.temp,
                           activity=s.cfg.activity, spin=s.cfg.spin,
                           precess=s.cfg.precess, seed=s.cfg.seed))
    for pl in sys.planets:
        p = orbit_position(pl.orbit, 0.0)
        v = orbit_velocity(pl.orbit, 0.0) * decay
        bodies.append(Body(kind=_PLANET, mass=0.02, radius=pl.radius,
                           pos=p.astype(np.float32), vel=v.astype(np.float32),
                           pcfg=pl.cfg))
    return bodies


def step(bodies: list, dt: float, g: float = 1.0):
    """One N-body step + mergers + swallows. Returns (bodies, events); each event
    is (world_pos, kind) with kind in {'merge','collapse','swallow'}."""
    alive = [b for b in bodies if b.alive]
    pos = np.array([b.pos for b in alive], np.float32)
    vel = np.array([b.vel for b in alive], np.float32)
    mass = np.array([b.mass for b in alive], np.float32)
    pos, vel = nbody_step(pos, vel, mass, dt, g=g)
    for i, b in enumerate(alive):
        b.pos, b.vel = pos[i], vel[i]

    events = []
    # black holes swallow anything inside ~their horizon
    for bh in [b for b in alive if b.alive and b.kind == BLACK_HOLE]:
        for b in alive:
            if b is bh or not b.alive or b.kind == BLACK_HOLE:
                continue
            if np.linalg.norm(b.pos - bh.pos) < bh.radius * 1.3:
                m = bh.mass + b.mass
                bh.vel = (bh.vel * bh.mass + b.vel * b.mass) / m
                bh.mass = m
                bh.radius = _bh_radius(m)
                b.alive = False
                events.append((b.pos.copy(), "swallow"))

    # star-star mergers
    stars = [b for b in alive if b.alive and not b.is_planet
             and b.kind != BLACK_HOLE]
    for ai in range(len(stars)):
        A = stars[ai]
        for bi in range(ai + 1, len(stars)):
            B = stars[bi]
            if not (A.alive and B.alive):
                continue
            if np.linalg.norm(A.pos - B.pos) < (A.radius + B.radius) * 0.8:
                m = A.mass + B.mass
                col = is_collapse(max(A.mass, B.mass), m)
                A.vel = (A.vel * A.mass + B.vel * B.mass) / m
                A.pos = (A.pos * A.mass + B.pos * B.mass) / m
                A.mass = m
                kind, radius, temp, act = _remnant_look(remnant_type(m), m)
                A.kind, A.radius, A.temp, A.activity = kind, radius, temp, act
                B.alive = False
                events.append((A.pos.copy(), "collapse" if col else "merge"))
    return [b for b in bodies if b.alive], events


_FLASH_COL = {"merge": np.array([1.0, 0.7, 0.35], np.float32),
              "collapse": np.array([0.8, 0.9, 1.0], np.float32),
              "swallow": np.array([1.0, 0.4, 0.2], np.float32)}
_FLASH_LIFE = 10


def _overlay_flashes(img, flashes, fidx, tmpl, W, H):
    eye, fwd, right, up, tanf, aspect = _cam_basis(tmpl, W, H)
    yy, xx = np.mgrid[0:H, 0:W]
    for (pos, kind, f0) in flashes:
        age = fidx - f0
        if age < 0 or age > _FLASH_LIFE:
            continue
        proj = _project(np.asarray(pos, np.float32), eye, fwd, right, up, tanf,
                        aspect, W, H)
        if proj is None:
            continue
        px, py, z = proj
        a = age / float(_FLASH_LIFE)
        big = 1.0 if kind in ("collapse",) else 0.5
        rad = (10.0 + 220.0 * a * big) * (30.0 / max(z, 1.0))
        amp = (1.0 - a) * (2.4 if kind == "collapse" else 1.3)
        d2 = (xx - px) ** 2 + (yy - py) ** 2
        glow = np.exp(-d2 / (rad * rad + 1e-3)) * amp
        img = img + glow[..., None] * _FLASH_COL[kind][None, None, :]
    return np.clip(img, 0.0, 1.0)


def simulate(sys: SystemConfig, frames: int = 60, dt: float = 0.05,
             width: int = 640, height: int = 400, g: float = 1.0,
             decay: float = 0.82, device: str = "cpu") -> list:
    """Run the destructive scenario and return `frames` composited images."""
    bodies = init_bodies(sys, decay=decay)
    flashes = []
    out = []
    for f in range(frames):
        bodies, events = step(bodies, dt, g=g)
        for (pos, kind) in events:
            flashes.append((pos, kind, f))
        star_bodies = [b for b in bodies if not b.is_planet]
        planet_bodies = [b for b in bodies if b.is_planet]
        stars = [Star(cfg=make_star(kind=b.kind, radius=b.radius, temp=b.temp,
                                    activity=b.activity, spin=b.spin,
                                    precess=b.precess, seed=b.seed),
                      orbit=Orbit(period=0.0)) for b in star_bodies]
        planets = [Planet(cfg=b.pcfg, orbit=Orbit(a=1.0, period=1.0),
                          radius=b.radius) for b in planet_bodies]
        sc = SystemConfig(stars=stars, planets=planets, nebula=sys.nebula,
                          dist=sys.dist, az=sys.az, el=sys.el, fov=sys.fov)
        spos = [b.pos for b in star_bodies]
        ppos = [b.pos for b in planet_bodies]
        img = render_system(sc, width, height, time=f * dt, device=device,
                            positions=(spos, ppos))
        img = _overlay_flashes(img, flashes, f, sys, width, height)
        out.append(img)
    return out
