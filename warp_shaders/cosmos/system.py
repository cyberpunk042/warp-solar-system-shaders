"""The configurable solar system — a layered, depth-composited renderer.

A `SystemConfig` is any mix of **stars** (1–7, each a `StarConfig` on an
`Orbit`) and **planets** (each a super-earth `PlanetConfig` on an `Orbit`), an
optional **nebula**, and a scenario (`stable` Kepler orbits, or `destructive`
N-body — see :mod:`.dynamics`).

Rendering is layered so it reuses every existing piece:

1. one kernel draws the **starfield + nebula + all stars** (spheres shaded by
   `bodies.shade_body`, plus coronas and pulsar beams) and writes a **depth**
   buffer (nearest star hit per pixel);
2. each **planet** is rendered centred by `superearth.render_planet` (lit toward
   the brightest star) and **billboarded** onto the scene at its projected
   screen position and size, depth-tested against the stars and the other
   planets so eclipses/transits order correctly;
3. an optional **black hole** lenses the whole composited scene (:mod:`.lensing`);
4. bloom + ACES tonemap once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..superearth.planet import PlanetConfig, render_planet
from .bodies import (BLACK_HOLE, NEUTRON, StarConfig, body_corona, neutron_axis,
                     pulsar_beams, shade_body)
from .nebula import nebula_march
from .orbits import Orbit, orbit_position

# apparent-brightness weight per kind, for choosing the dominant light source
_LUM = {0: 1.0, 1: 1.3, 2: 0.5, 3: 0.0}       # sun / neutron / dwarf / black hole


@dataclass
class Star:
    cfg: StarConfig
    orbit: Orbit = field(default_factory=lambda: Orbit(period=0.0))


@dataclass
class Planet:
    cfg: PlanetConfig
    orbit: Orbit
    radius: float = 0.6
    quality: str = "medium"
    seed_time: float = 0.0


@dataclass
class Nebula:
    center: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.0                        # 0 => no nebula
    seed: float = 2.0


@dataclass
class SystemConfig:
    stars: List[Star]
    planets: List[Planet] = field(default_factory=list)
    nebula: Optional[Nebula] = None
    scenario: str = "stable"                   # stable | destructive
    dist: float = 16.0
    az: float = 0.6
    el: float = 0.22
    fov: float = 40.0
    tscale: float = 1.0


# --------------------------------------------------------------------------- #
# star + nebula + starfield kernel                                            #
# --------------------------------------------------------------------------- #

@wp.kernel
def _stars_kernel(img: wp.array2d(dtype=wp.vec3), depth: wp.array2d(dtype=float),
                  cam: Camera, centers: wp.array(dtype=wp.vec3),
                  radii: wp.array(dtype=float), kinds: wp.array(dtype=int),
                  temps: wp.array(dtype=float), acts: wp.array(dtype=float),
                  spins: wp.array(dtype=float), precs: wp.array(dtype=float),
                  seeds: wp.array(dtype=float), nstars: int,
                  neb_center: wp.vec3, neb_radius: float, neb_seed: float,
                  neb_steps: int, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    col = stars(rd)
    if neb_radius > 0.0:
        nv = nebula_march(ro, rd, neb_center, neb_radius, neb_seed, time, neb_steps)
        col = col * nv[3] + wp.vec3(nv[0], nv[1], nv[2])

    best_t = float(1.0e30)
    for k in range(nstars):
        c = centers[k]
        R = radii[k]
        oc = ro - c
        b = wp.dot(oc, rd)
        cc = wp.dot(oc, oc) - R * R
        disc = b * b - cc
        if disc > 0.0:
            t = -b - wp.sqrt(disc)
            if t > 0.0 and t < best_t:
                best_t = t
                cfg = StarConfig()
                cfg.kind = kinds[k]
                cfg.radius = R
                cfg.temp = temps[k]
                cfg.activity = acts[k]
                cfg.spin = spins[k]
                cfg.precess = precs[k]
                cfg.seed = seeds[k]
                dir = wp.normalize((ro + rd * t) - c)
                col = shade_body(dir, rd, cfg, time)

    # additive glows (corona + pulsar beams) from every star
    for k in range(nstars):
        c = centers[k]
        R = radii[k]
        cfg = StarConfig()
        cfg.kind = kinds[k]
        cfg.radius = R
        cfg.temp = temps[k]
        cfg.activity = acts[k]
        cfg.spin = spins[k]
        cfg.precess = precs[k]
        cfg.seed = seeds[k]
        oc = ro - c
        tca = -wp.dot(oc, rd)
        if tca > 0.0:
            d = wp.length(oc + rd * tca)
            col = col + body_corona(d / R, cfg, time)
        if kinds[k] == NEUTRON:
            ax = neutron_axis(cfg, time)
            col = col + pulsar_beams(oc, rd, ax, cfg, time)

    depth[i, j] = best_t
    img[i, j] = col


# --------------------------------------------------------------------------- #
# host helpers                                                                #
# --------------------------------------------------------------------------- #

def _cam_basis(sys: SystemConfig, width, height):
    d, az, el = sys.dist, sys.az, sys.el
    eye = np.array([d * math.cos(el) * math.sin(az), d * math.sin(el),
                    d * math.cos(el) * math.cos(az)], np.float32)
    fwd = -eye / (np.linalg.norm(eye) + 1e-9)
    right = np.cross(fwd, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(right, fwd)
    tanf = math.tan(math.radians(sys.fov) * 0.5)
    aspect = width / height
    return eye, fwd, right, up, tanf, aspect


def _project(P, eye, fwd, right, up, tanf, aspect, W, H):
    rel = P - eye
    z = float(rel @ fwd)
    if z <= 1e-3:
        return None
    x = float(rel @ right)
    y = float(rel @ up)
    u = (x / z) / (aspect * tanf)
    v = (y / z) / tanf
    px = (u + 1.0) * 0.5 * W - 0.5
    py = H - 0.5 - (v + 1.0) * 0.5 * H
    return px, py, z


def _azel(d):
    d = d / (np.linalg.norm(d) + 1e-9)
    el = math.asin(float(np.clip(d[1], -1.0, 1.0)))
    az = math.atan2(float(d[0]), float(d[2]))
    return az, el


# render_planet's centred disk radius as a fraction of the buffer half-height,
# for dist=3.4, fov=42 (asin(R/dist) mapped through tan_half_fov)
_BB_DIST, _BB_FOV = 3.4, 42.0
_DISK_FRAC = math.tan(math.asin(1.0 / _BB_DIST)) / math.tan(math.radians(_BB_FOV) * 0.5)

# render_planet's FIXED camera basis (az=0.6, el=0.28) — the billboard always
# shows the planet from this view, so the sun must be placed relative to it
_RP_AZ, _RP_EL = 0.6, 0.28
_RP_EYE = np.array([math.cos(_RP_EL) * math.sin(_RP_AZ), math.sin(_RP_EL),
                    math.cos(_RP_EL) * math.cos(_RP_AZ)], np.float32)
_RP_FWD = -_RP_EYE / np.linalg.norm(_RP_EYE)
_RP_RIGHT = np.cross(_RP_FWD, np.array([0, 1, 0], np.float32))
_RP_RIGHT /= np.linalg.norm(_RP_RIGHT)
_RP_UP = np.cross(_RP_RIGHT, _RP_FWD)
_RP_VIEW = -_RP_FWD                                  # planet -> camera


def _planet_sun(P, spos, eye, proj_planet, proj_star):
    """Sun az/el (in render_planet's frame) that lights the planet's disk toward
    the star's on-screen direction, with the correct day/night phase."""
    realstar = spos - P
    realview = eye - P
    ns = np.linalg.norm(realstar) + 1e-9
    nv = np.linalg.norm(realview) + 1e-9
    cosph = float(np.clip((realstar @ realview) / (ns * nv), -0.985, 0.985))
    sinph = math.sqrt(max(1.0 - cosph * cosph, 0.0))
    dsx = proj_star[0] - proj_planet[0]
    dsy = -(proj_star[1] - proj_planet[1])           # flip to y-up
    n = math.hypot(dsx, dsy)
    if n < 1e-6:
        dsx, dsy = 1.0, 0.0
    else:
        dsx, dsy = dsx / n, dsy / n
    sun_rp = _RP_VIEW * cosph + (_RP_RIGHT * dsx + _RP_UP * dsy) * sinph
    sun_rp = sun_rp / (np.linalg.norm(sun_rp) + 1e-9)
    az = math.atan2(float(sun_rp[0]), float(sun_rp[2]))
    el = math.asin(float(np.clip(sun_rp[1], -1.0, 1.0)))
    return az, el


def render_system(sys: SystemConfig, width: int, height: int, time: float = 0.0,
                  device: str = "cpu", positions=None) -> np.ndarray:
    """Render the system at `time` to an ``(H, W, 3)`` image.

    `positions` optionally supplies precomputed (star_pos, planet_pos) world
    arrays (used by the destructive N-body driver); otherwise Kepler orbits are
    evaluated at ``time * tscale``.
    """
    t = time * sys.tscale
    eye, fwd, right, up, tanf, aspect = _cam_basis(sys, width, height)
    cam = make_camera(tuple(eye), (0.0, 0.0, 0.0), fov_deg=sys.fov,
                      aspect=aspect)

    if positions is not None:
        star_pos, planet_pos = positions
    else:
        star_pos = [orbit_position(s.orbit, t) for s in sys.stars]
        planet_pos = [orbit_position(p.orbit, t) for p in sys.planets]

    ns = max(len(sys.stars), 1)
    centers = np.zeros((ns, 3), np.float32)
    radii = np.ones(ns, np.float32)
    kinds = np.zeros(ns, np.int32)
    temps = np.full(ns, 0.5, np.float32)
    acts = np.full(ns, 0.5, np.float32)
    spins = np.ones(ns, np.float32)
    precs = np.zeros(ns, np.float32)
    seeds = np.ones(ns, np.float32)
    for k, s in enumerate(sys.stars):
        centers[k] = star_pos[k]
        radii[k] = s.cfg.radius
        kinds[k] = s.cfg.kind
        temps[k] = s.cfg.temp
        acts[k] = s.cfg.activity
        spins[k] = s.cfg.spin
        precs[k] = s.cfg.precess
        seeds[k] = s.cfg.seed

    neb = sys.nebula
    nc = neb.center if neb else (0.0, 0.0, 0.0)
    nr = neb.radius if neb else 0.0
    nsd = neb.seed if neb else 0.0

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    depth = wp.zeros((height, width), dtype=float, device=device)
    wp.launch(_stars_kernel, dim=(height, width),
              inputs=[img, depth, cam,
                      wp.array(centers, dtype=wp.vec3, device=device),
                      wp.array(radii, dtype=float, device=device),
                      wp.array(kinds, dtype=int, device=device),
                      wp.array(temps, dtype=float, device=device),
                      wp.array(acts, dtype=float, device=device),
                      wp.array(spins, dtype=float, device=device),
                      wp.array(precs, dtype=float, device=device),
                      wp.array(seeds, dtype=float, device=device),
                      int(len(sys.stars)),
                      wp.vec3(nc[0], nc[1], nc[2]), float(nr), float(nsd), 56,
                      float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    scene = img.numpy()
    zbuf = depth.numpy()

    # planets, far -> near, billboarded and depth-tested
    star_lum = [(_LUM.get(s.cfg.kind, 0.0), star_pos[k])
                for k, s in enumerate(sys.stars)]
    order = sorted(range(len(sys.planets)),
                   key=lambda k: -float((planet_pos[k] - eye) @ fwd))
    for k in order:
        pl = sys.planets[k]
        P = planet_pos[k]
        proj = _project(P, eye, fwd, right, up, tanf, aspect, width, height)
        if proj is None:
            continue
        px, py, z = proj
        rp = pl.radius * ((height * 0.5) / tanf) / z      # projected pixel radius
        if rp < 1.0:
            continue
        S = int(min(max(2.0 * rp / _DISK_FRAC * 0.5, 12), max(width, height)))
        # light from the brightest star, placed for the correct on-screen phase
        if star_lum:
            _, spos = max(star_lum, key=lambda a: a[0])
            sproj = _project(spos, eye, fwd, right, up, tanf, aspect, width, height)
            if sproj is None:
                saz, sel = _azel(spos - P)
            else:
                saz, sel = _planet_sun(P, spos, eye, (px, py), (sproj[0], sproj[1]))
        else:
            saz, sel = 1.0, 0.35
        pbuf = render_planet(pl.cfg, S, S, time=t + pl.seed_time,
                             device=device, quality=pl.quality,
                             sun_az=saz, sun_el=sel, dist=_BB_DIST, fov=_BB_FOV,
                             relief=False)
        _composite_billboard(scene, zbuf, pbuf, px, py, z, S)

    hdr = post.bloom(scene, threshold=1.5, strength=0.4,
                     radius=max(3, int(min(width, height) * 0.02)), passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.03)


def _composite_billboard(scene, zbuf, pbuf, px, py, z, S):
    """Alpha-composite a centred planet buffer `pbuf` (SxS) onto `scene` at
    screen (px, py), masking to the planet disk+atmosphere and depth-testing
    against `zbuf` (nearer wins)."""
    H, W = scene.shape[:2]
    disk_r = _DISK_FRAC * S * 0.5                       # planet radius in buffer px
    x0 = int(round(px - S * 0.5))
    y0 = int(round(py - S * 0.5))
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(W, x0 + S), min(H, y0 + S)
    if sx1 <= sx0 or sy1 <= sy0:
        return
    ys = np.arange(sy0, sy1)
    xs = np.arange(sx0, sx1)
    by = ys - y0
    bx = xs - x0
    cx = S * 0.5 - 0.5
    cy = S * 0.5 - 0.5
    rr = np.sqrt((bx[None, :] - cx) ** 2 + (by[:, None] - cy) ** 2)
    alpha = np.clip((disk_r * 1.14 - rr) / (disk_r * 0.14 + 1e-6), 0.0, 1.0)
    vis = (z < zbuf[sy0:sy1, sx0:sx1]) & (alpha > 0.0)
    a = (alpha * vis).astype(np.float32)[..., None]
    sub = pbuf[by[0]:by[0] + (sy1 - sy0), bx[0]:bx[0] + (sx1 - sx0)]
    scene[sy0:sy1, sx0:sx1] = scene[sy0:sy1, sx0:sx1] * (1.0 - a) + sub * a
    zb = zbuf[sy0:sy1, sx0:sx1]
    zb[vis] = z
    zbuf[sy0:sy1, sx0:sx1] = zb
