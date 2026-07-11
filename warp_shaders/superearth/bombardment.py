"""Nuclear bombardment of the super-earth — configurable, and pimped.

A planet being bombed from orbit: you set the **amount** of warheads, the
**distribution formula** over the globe, the **delay** before the first strike,
the **interval** between waves, and how many go off **in parallel** per wave.
Each detonation flashes, throws an expanding fireball, and leaves a glowing
crater scar. Built on the engine's :class:`~warp_shaders.sim.engine.ParticleSystem`,
composited over the ray-marched planet with a front-hemisphere occlusion cull so
strikes only show on the visible face.

The blast is "pimped" vs the flat sim: radial ejecta off the sphere, a hot
blackbody flash core, an expanding shock-ring scar on the surface, and lingering
embers that cool through the blackbody ramp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..engine import post
from ..sim.engine import _RAMP_B, _RAMP_G, _RAMP_R, _STOPS, ParticleSystem
from .planet import render_planet


@dataclass
class BombConfig:
    n: int = 30                 # number of warheads
    delay: float = 0.4          # seconds before the first strike
    interval: float = 0.3       # seconds between waves
    parallel: int = 4           # detonations per wave
    formula: str = "clustered"  # uniform | clustered | equatorial | spiral
    yield_scale: float = 1.0
    seed: int = 1


def _rand_dir(rng):
    v = rng.normal(size=3)
    return v / (np.linalg.norm(v) + 1e-9)


def _cap_dir(rng, axis, min_cos):
    """A random unit vector within an angular cap around `axis` (dot >= min_cos)."""
    axis = np.asarray(axis, np.float32)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    for _ in range(64):
        v = _rand_dir(rng)
        if float(v @ axis) >= min_cos:
            return v.astype(np.float32)
    return axis


def sites(n: int, formula: str, seed: int, front=None) -> np.ndarray:
    """`n` unit vectors on the globe following a distribution formula.

    When ``front`` (a unit direction toward the camera) is given, strikes are
    biased onto the visible hemisphere so the bombardment clearly lands on the
    face we see rather than drifting onto the limb.
    """
    rng = np.random.default_rng(seed)
    p = np.zeros((n, 3), np.float32)
    g = 2.399963
    if formula == "uniform":
        for k in range(n):
            y = 1.0 - 2.0 * (k + 0.5) / n
            r = math.sqrt(max(1.0 - y * y, 0.0))
            p[k] = (r * math.cos(g * k), y, r * math.sin(g * k))
    elif formula == "equatorial":
        for k in range(n):
            th = rng.uniform(0, 2 * math.pi)
            y = rng.uniform(-0.28, 0.28)
            r = math.sqrt(max(1.0 - y * y, 0.0))
            p[k] = (r * math.cos(th), y, r * math.sin(th))
    elif formula == "spiral":
        for k in range(n):
            t = k / max(n - 1, 1)
            y = 1.0 - 2.0 * t
            r = math.sqrt(max(1.0 - y * y, 0.0))
            th = k * g * 2.6
            p[k] = (r * math.cos(th), y, r * math.sin(th))
    else:  # clustered — a few strike zones (like real arsenals)
        nc = max(2, n // 6)
        if front is not None:
            centers = [_cap_dir(rng, front, 0.45) for _ in range(nc)]
        else:
            centers = [_rand_dir(rng) for _ in range(nc)]
        for k in range(n):
            c = centers[int(rng.integers(nc))]
            d = c + rng.normal(scale=0.22, size=3)
            p[k] = d / (np.linalg.norm(d) + 1e-9)
        return p
    # point formulas: fold any back-facing strike onto the visible cap
    if front is not None:
        f = np.asarray(front, np.float32)
        f = f / (np.linalg.norm(f) + 1e-9)
        for k in range(n):
            if float(p[k] @ f) < 0.2:
                p[k] = _cap_dir(rng, f, 0.3)
    return p


def _fire_frames(bcfg: BombConfig, dt: float):
    par = max(bcfg.parallel, 1)
    return [int(round((bcfg.delay + (i // par) * bcfg.interval) / dt))
            for i in range(bcfg.n)]


def _spawn_blast(ps, site, count, rng):
    base = np.asarray(site, np.float32)
    for _ in range(count):
        # bias ejecta radially outward from the impact so the fireball plumes
        # up off the surface instead of streaking sideways across the limb
        d = 1.4 * base + rng.normal(scale=0.4, size=3).astype(np.float32)
        d /= np.linalg.norm(d) + 1e-9
        speed = rng.uniform(0.1, 0.5)
        # spawn hot but not pure-white: the blackbody ramp reads orange->white
        # across 0.55..0.9, giving a fireball with colour instead of a flat disc
        ps.spawn(pos=(base * 1.01).tolist(),
                 vel=(d * speed).tolist(),
                 temp=float(rng.uniform(0.55, 0.9)),
                 life=float(rng.uniform(0.4, 1.2)), kind=2)


def _splat(width, height, ps, eye, fov, scars):
    """Blackbody particle splat + shock-ring scars, culling the far hemisphere."""
    frame = np.zeros((height, width, 3), np.float32)
    eye = np.asarray(eye, np.float32)
    eye_hat = eye / (np.linalg.norm(eye) + 1e-9)
    fwd = -eye_hat
    right = np.cross(fwd, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)
    f = (height * 0.5) / math.tan(math.radians(fov) * 0.5)

    def project(P):
        rel = P - eye
        cz = rel @ fwd
        cx = rel @ right
        cy = rel @ up
        px = width * 0.5 + (cx / cz) * f
        py = height * 0.5 - (cy / cz) * f
        return px, py, cz

    # shock-ring scars on the surface (smooth expanding rings, age-coloured)
    for (c, radius, glow, rcol) in scars:
        cn = c / (np.linalg.norm(c) + 1e-9)
        if float(cn @ eye_hat) < 0.15:            # on the far side -> hidden
            continue
        # sample the ring as points on the sphere at angular `radius` around cn
        t1 = np.cross(cn, np.array([0, 1, 0], np.float32))
        if np.linalg.norm(t1) < 1e-3:
            t1 = np.cross(cn, np.array([1, 0, 0], np.float32))
        t1 /= np.linalg.norm(t1) + 1e-9
        t2 = np.cross(cn, t1)
        # dense enough that the projected circle reads as a continuous line
        m = max(64, int(220 * max(radius, 0.05)))
        ang = np.linspace(0, 2 * math.pi, m, dtype=np.float32)
        ring = (math.cos(radius) * cn[None, :]
                + math.sin(radius) * (np.cos(ang)[:, None] * t1[None, :]
                                      + np.sin(ang)[:, None] * t2[None, :]))
        vis = (ring / np.linalg.norm(ring, axis=1, keepdims=True)) @ eye_hat > 0.1
        rp = ring[vis] * 1.01
        if len(rp) == 0:
            continue
        px, py, cz = project(rp)
        col = (rcol * glow).astype(np.float32)
        pxi = np.round(px).astype(np.int64)
        pyi = np.round(py).astype(np.int64)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                w = 1.0 if (dx == 0 and dy == 0) else 0.5
                xx = pxi + dx
                yy = pyi + dy
                ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                if ok.any():
                    np.add.at(frame, (yy[ok], xx[ok]), col * w)

    # particles
    mask = ps.alive_mask()
    if mask.any():
        P = ps.pos[mask]
        temp = ps.temp[mask]
        pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        vis = (pn @ eye_hat > 0.12)               # cull occluded far side
        P, temp = P[vis], temp[vis]
        if len(P):
            px, py, cz = project(P)
            infront = cz > 0.05
            px, py, temp = px[infront], py[infront], temp[infront]
            t = np.clip(temp, 0.0, 1.0)
            col = np.stack([np.interp(t, _STOPS, _RAMP_R),
                            np.interp(t, _STOPS, _RAMP_G),
                            np.interp(t, _STOPS, _RAMP_B)], 1).astype(np.float32)
            bright = (0.25 + 0.85 * t).astype(np.float32)
            contrib = col * bright[:, None]
            pxi = np.round(px).astype(np.int64)
            pyi = np.round(py).astype(np.int64)
            r = 3
            s2 = 2.0 * (r * 0.5) ** 2
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    w = math.exp(-(dx * dx + dy * dy) / s2)
                    xx = pxi + dx
                    yy = pyi + dy
                    ok = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
                    if ok.any():
                        np.add.at(frame, (yy[ok], xx[ok]), contrib[ok] * w)
    return frame


def _eye_for(dist):
    # must match render_planet's camera (az=0.6, el=0.28, mouse=0)
    az, el = 0.6, 0.28
    return np.array([dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                     dist * math.cos(el) * math.cos(az)], np.float32)


def _roty(v, a):
    """Rotate `v` about +Y by `a` — matches render_kernel's ray rotation."""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([ca * v[0] - sa * v[2], v[1], sa * v[0] + ca * v[2]],
                    np.float32)


def run(planet_cfg, bcfg: BombConfig, width, height, frames, dt, device,
        dist, fov, moons=None, quality="low"):
    """Render the whole bombardment; returns a list of `frames` composited images."""
    base_eye = _eye_for(dist)
    front = base_eye / (np.linalg.norm(base_eye) + 1e-9)
    spin = float(getattr(planet_cfg, "spin", 0.0))
    rng = np.random.default_rng(bcfg.seed)
    st = sites(bcfg.n, bcfg.formula, bcfg.seed, front=front)
    ff = _fire_frames(bcfg, dt)
    ps = ParticleSystem(int(bcfg.n * 260 * bcfg.yield_scale) + 800, device)
    scar_state = [dict(active=False, t0=0) for _ in range(bcfg.n)]
    out = []
    for fidx in range(frames):
        for i in range(bcfg.n):
            if ff[i] == fidx:
                _spawn_blast(ps, st[i], int(200 * bcfg.yield_scale), rng)
                scar_state[i] = dict(active=True, t0=fidx)
        ps.step(dt, g=0.0, buoy=0.0, drag=1.5, cool=0.7, vortex=0,
                cap_y=0.0, ring_a=1.0)
        # active scars: expanding, fading rings
        scars = []
        white_hot = np.array([1.0, 0.95, 0.85], np.float32)
        ember = np.array([1.0, 0.42, 0.10], np.float32)
        for i in range(bcfg.n):
            if scar_state[i]["active"]:
                age = (fidx - scar_state[i]["t0"]) * dt
                radius = 0.04 + age * 0.5
                glow = max(0.0, 1.0 - age * 1.1)
                if glow <= 0.02 or radius > 1.2:
                    scar_state[i]["active"] = False
                else:
                    warm = min(1.0, age * 1.4)     # young ring hot-white -> orange
                    rcol = white_hot * (1.0 - warm) + ember * warm
                    scars.append((st[i].astype(np.float32), radius, glow, rcol))
        globe = render_planet(planet_cfg, width, height, time=fidx * dt,
                              device=device, quality=quality, moons=moons,
                              dist=dist, fov=fov, relief=False)
        # match the kernel's per-frame planet spin (rays rotate by -time*spin)
        eye = _roty(base_eye, -fidx * dt * spin)
        parts = _splat(width, height, ps, eye, fov, scars)
        comp = globe + parts
        comp = post.bloom(comp, threshold=1.35, strength=0.35,
                          radius=max(3, int(min(width, height) * 0.02)), passes=3)
        out.append(np.clip(comp, 0.0, 1.0))
    return out
