"""World-space celestial-body shaders — the star library for the solar system.

Every body is a sphere at an arbitrary world center, shaded emissively (stars
emit; they are not PBR-lit) so it composites into a scene by depth. One kernel
branches on ``StarConfig.kind``:

    0  sun          — granulated photosphere, limb darkening, corona, spots
    1  neutron star — tiny, blistering blue-white, twin pulsar beams, precession
    2  white dwarf  — small, hot, smooth degenerate surface, tight glow

(The black hole, kind 3, lenses light and lives in :mod:`.blackhole`.)

The reusable pieces are `@wp.func`s (`shade_body`, `body_corona`, `_temp_color`,
`_beam`) so the composited solar-system renderer can call them per pixel; the
`render_star` host helper renders one centred body for standalone verification.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import (domain_warp3, fbm3, ridged3, worley3,
                                 worley3_f2)

# body kinds
SUN = 0
NEUTRON = 1
WHITE_DWARF = 2
BLACK_HOLE = 3


@wp.struct
class StarConfig:
    kind: int         # SUN / NEUTRON / WHITE_DWARF / BLACK_HOLE
    radius: float     # world radius
    temp: float       # colour temperature 0 (cool red) .. 1 (hot blue)
    activity: float   # surface turbulence, flares, beam strength
    spin: float       # rotation / animation rate
    precess: float    # beam-axis precession rate (neutron star)
    seed: float


# --------------------------------------------------------------------------- #
# colour                                                                       #
# --------------------------------------------------------------------------- #

@wp.func
def _temp_color(t: float) -> wp.vec3:
    """A blackbody-ish ramp: red → orange → yellow → white → blue-white."""
    t = wp.clamp(t, 0.0, 1.0)
    red = wp.vec3(1.0, 0.30, 0.06)
    orange = wp.vec3(1.0, 0.60, 0.22)
    yellow = wp.vec3(1.0, 0.92, 0.70)
    white = wp.vec3(1.0, 1.0, 0.97)
    blue = wp.vec3(0.72, 0.83, 1.0)
    if t < 0.25:
        k = t / 0.25
        return red * (1.0 - k) + orange * k
    if t < 0.5:
        k = (t - 0.25) / 0.25
        return orange * (1.0 - k) + yellow * k
    if t < 0.75:
        k = (t - 0.5) / 0.25
        return yellow * (1.0 - k) + white * k
    k = (t - 0.75) / 0.25
    return white * (1.0 - k) + blue * k


# --------------------------------------------------------------------------- #
# surfaces                                                                     #
# --------------------------------------------------------------------------- #

@wp.func
def _shade_sun(dir: wp.vec3, rd: wp.vec3, cfg: StarConfig, time: float) -> wp.vec3:
    s = wp.vec3(cfg.seed, cfg.seed * 1.7, cfg.seed * 2.3)
    w = time * 0.06 * (0.5 + cfg.spin)
    drift = wp.vec3(w, w * 0.3, 0.0)
    # convection granulation: bright polygonal cells with dark boundary lanes.
    # worley F2-F1 is ~0 at cell boundaries and large in the interior, so it
    # lights the CELLS (not the centres) — real granulation, not sparkly dots
    wd = domain_warp3(dir * 3.0 + s + drift, 3, 0.6)
    ed = worley3_f2(dir * 8.0 + s + drift * 2.0 + wp.vec3(wd, 0.0, 0.0))
    gran = wp.smoothstep(0.0, 0.2, ed[1] - ed[0])        # 1 in cell, 0 in lane (soft)
    turb = fbm3(dir * 6.0 + s + drift * 1.5, 5) - 0.5
    # golden granule cores over deep-orange lanes — the colour must survive the
    # tonemap (red-dominant, low blue) instead of clipping to white
    hot = _temp_color(wp.clamp(cfg.temp - 0.12, 0.0, 1.0))
    cool = _temp_color(wp.clamp(cfg.temp - 0.34, 0.0, 1.0)) * 0.48
    col = cool * (1.0 - gran) + hot * gran
    col = col * (0.85 + 0.4 * turb * (0.5 + cfg.activity))
    # fine mottle so the coarse cells still carry texture
    mott = fbm3(dir * 22.0 + s + drift * 3.0, 3) - 0.5
    col = col * (0.9 + 0.22 * mott)
    # sunspots — cool dark patches (umbra + penumbra)
    spot = wp.smoothstep(0.60, 0.72, fbm3(dir * 3.5 + s + wp.vec3(3.0, 0.0, 0.0), 4))
    col = col * (1.0 - 0.85 * spot * (0.4 + 0.6 * cfg.activity))
    # limb darkening — the disk edge is dimmer + redder (grazing optical depth)
    ndv = wp.max(wp.dot(dir, -rd), 0.0)
    limb = 0.32 + 0.68 * wp.pow(ndv, 0.5)
    col = wp.cw_mul(col, wp.vec3(limb, limb * 0.92, limb * 0.78))
    return col * (1.35 + 0.45 * cfg.activity)            # HDR emissive for bloom


@wp.func
def _beam(dir: wp.vec3, axis: wp.vec3, width: float) -> float:
    """A twin polar beam: bright along ±axis, falling off with angle."""
    c = wp.abs(wp.dot(dir, axis))
    return wp.pow(wp.clamp(c, 0.0, 1.0), width)


@wp.func
def _shade_neutron(dir: wp.vec3, rd: wp.vec3, cfg: StarConfig,
                   time: float) -> wp.vec3:
    s = wp.vec3(cfg.seed, cfg.seed * 2.1, cfg.seed * 1.3)
    # blistering, near-smooth surface with faint hotspots
    tex = fbm3(dir * 20.0 + s + wp.vec3(time * 0.4, 0.0, 0.0), 4)
    hot = 0.9 + 0.5 * (tex - 0.5)
    base = _temp_color(wp.clamp(cfg.temp, 0.7, 1.0))     # always blue-white hot
    col = base * hot
    # magnetic-pole hotspots — the beam roots glow hardest
    ax = wp.normalize(wp.vec3(wp.sin(time * cfg.precess) * 0.5, 1.0,
                              wp.cos(time * cfg.precess) * 0.35))
    caps = _beam(dir, ax, 6.0)
    col = col + base * caps * (1.5 + cfg.activity)
    ndv = wp.max(wp.dot(dir, -rd), 0.0)
    limb = 0.5 + 0.5 * wp.pow(ndv, 0.6)
    col = col * limb
    return col * (2.4 + 1.4 * cfg.activity)            # bright, compact body


@wp.func
def _shade_white_dwarf(dir: wp.vec3, rd: wp.vec3, cfg: StarConfig,
                       time: float) -> wp.vec3:
    s = wp.vec3(cfg.seed, cfg.seed * 1.9, cfg.seed * 2.7)
    # smooth, faintly mottled degenerate surface, blue-white hot
    tex = fbm3(dir * 10.0 + s + wp.vec3(time * 0.05, 0.0, 0.0), 4) - 0.5
    base = wp.cw_mul(_temp_color(wp.clamp(cfg.temp, 0.6, 0.95)),
                     wp.vec3(0.86, 0.92, 1.05))         # push toward blue
    col = base * (1.0 + 0.22 * tex)
    ndv = wp.max(wp.dot(dir, -rd), 0.0)
    limb = 0.5 + 0.5 * wp.pow(ndv, 0.55)
    col = col * limb
    return col * (1.7 + cfg.activity)


@wp.func
def neutron_axis(cfg: StarConfig, time: float) -> wp.vec3:
    """The (precessing) magnetic / beam axis of a neutron star."""
    a = time * cfg.precess
    return wp.normalize(wp.vec3(wp.sin(a) * 0.5, 1.0, wp.cos(a) * 0.35))


@wp.func
def pulsar_beams(ro: wp.vec3, rd: wp.vec3, axis: wp.vec3, cfg: StarConfig,
                 time: float) -> wp.vec3:
    """Twin polar beams as narrow cones of light along ±axis through the star.
    Exact ray-vs-axis-line closest distance, faded along the beam length."""
    b = wp.dot(rd, axis)
    denom = 1.0 - b * b
    if denom < 1.0e-4:                                   # ray parallel to axis
        return wp.vec3(0.0, 0.0, 0.0)
    d = wp.dot(rd, ro)
    e = wp.dot(axis, ro)
    t = (b * e - d) / denom                              # param along the ray
    u = (e - b * d) / denom                              # param along the axis
    if t <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    perp = wp.length((ro + rd * t) - axis * u)
    R = cfg.radius
    width = (0.10 * R + 0.015) * (1.0 + 0.18 * wp.abs(u) / wp.max(R, 1e-3))
    fall = wp.exp(-(perp / width) * (perp / width))
    lenf = wp.exp(-wp.abs(u) / (9.0 * R))                # fade along the jet
    inten = fall * lenf * (1.6 + 1.4 * cfg.activity)
    return wp.vec3(0.62, 0.80, 1.0) * inten             # electric blue-white


@wp.func
def shade_body(dir: wp.vec3, rd: wp.vec3, cfg: StarConfig, time: float) -> wp.vec3:
    """Emissive surface colour of body `cfg` at surface direction `dir`."""
    if cfg.kind == BLACK_HOLE:
        return wp.vec3(0.0, 0.0, 0.0)      # drawn by the lensing pass, not here
    if cfg.kind == NEUTRON:
        return _shade_neutron(dir, rd, cfg, time)
    if cfg.kind == WHITE_DWARF:
        return _shade_white_dwarf(dir, rd, cfg, time)
    return _shade_sun(dir, rd, cfg, time)


@wp.func
def body_corona(d_over_r: float, cfg: StarConfig, time: float) -> wp.vec3:
    """Corona / glow halo for a ray whose closest approach is `d_over_r`×radius
    from the centre (>1 = outside the disk). Returns additive HDR colour."""
    if d_over_r < 1.0 or cfg.kind == BLACK_HOLE:
        return wp.vec3(0.0, 0.0, 0.0)
    base = _temp_color(cfg.temp)
    if cfg.kind == NEUTRON:
        base = _temp_color(wp.clamp(cfg.temp, 0.7, 1.0))
    # inverse-power falloff, with a hard-ish cutoff so the halo doesn't gamma-lift
    # into a frame-wide grey wash (bloom supplies the wide, soft glow instead)
    p = 3.4
    amp = 0.55 + 0.45 * cfg.activity
    cut = 2.6
    if cfg.kind == NEUTRON:
        p = 3.8
        amp = 1.1
        cut = 2.2
    if cfg.kind == WHITE_DWARF:
        p = 3.4
        amp = 0.8
        cut = 2.3
    g = wp.pow(1.0 / d_over_r, p) * wp.smoothstep(cut, 1.0, d_over_r)
    return base * (g * amp)


# --------------------------------------------------------------------------- #
# single-body kernel + host helper (standalone verification)                  #
# --------------------------------------------------------------------------- #

@wp.kernel
def star_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, cfg: StarConfig,
                time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    col = stars(rd)
    R = cfg.radius
    b = wp.dot(ro, rd)
    c = wp.dot(ro, ro) - R * R
    disc = b * b - c
    tca = -b
    hit = int(0)
    if disc > 0.0:
        t = -b - wp.sqrt(disc)
        if t > 0.0:
            p = ro + rd * t
            dir = wp.normalize(p)
            col = shade_body(dir, rd, cfg, time)
            hit = 1
    if hit == 0 and tca > 0.0:
        # closest approach distance of the ray line to the centre
        d = wp.sqrt(wp.max(wp.dot(ro, ro) - tca * tca, 0.0))
        col = col + body_corona(d / R, cfg, time)
    # pulsar beams shine past the surface, so add them for every ray
    if cfg.kind == NEUTRON:
        ax = neutron_axis(cfg, time)
        col = col + pulsar_beams(ro, rd, ax, cfg, time)
    img[i, j] = col


def render_star(cfg: StarConfig, width: int, height: int, time: float = 0.0,
                device: str = "cpu", dist: float = 4.0, fov: float = 42.0,
                bloom_strength: float = 0.7) -> np.ndarray:
    """Render one centred body to an ``(H, W, 3)`` image (for verification)."""
    eye = (dist * 0.15, dist * 0.12, dist)
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=fov, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(star_kernel, dim=(height, width),
              inputs=[img, cam, cfg, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=bloom_strength, radius=r,
                     passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


def make_star(kind=SUN, radius=1.0, temp=0.5, activity=0.6, spin=1.0,
              precess=0.0, seed=1.0) -> StarConfig:
    cfg = StarConfig()
    cfg.kind = int(kind)
    cfg.radius = float(radius)
    cfg.temp = float(temp)
    cfg.activity = float(activity)
    cfg.spin = float(spin)
    cfg.precess = float(precess)
    cfg.seed = float(seed)
    return cfg
