"""The configurable super-earth planet — a heightfield sphere in one Warp kernel.

Everything is a knob on :class:`PlanetConfig` (a ``@wp.struct`` of floats/ints so
it rides into the kernel): turn each feature on or off independently. The surface
is a **displaced sphere** ray-marched per pixel — real relief on the limb — shaded
with the engine's PBR, over an optional atmosphere and starfield.

Scale is normalised (radius ~1) so features compose cleanly and the atmosphere is
this module's own (controllable), not the earth-scale LUT path.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import domain_warp3, fbm3, ridged3, worley3, worley3_f2

_R = 1.0                      # planet radius (normalised)
_RS = 0.055                   # relief scale (fraction of radius)
_GOLDEN = 2.399963


@wp.struct
class PlanetConfig:
    seed: float
    # terrain
    mountain: float           # mountain amplitude (0 = smooth)
    sea_level: float          # ocean threshold in elevation units
    has_ocean: int
    has_lakes: int
    has_rivers: int
    snow: float               # snow-cap amount
    # volcanism
    has_volcano: int
    volcano_n: int
    lava: float               # lava glow amount (0 = cold)
    # life
    veg: float                # vegetation amount (0 = barren)
    alive: float              # bioluminescence (0 = not alive)
    city: float               # night-side city lights
    # sky
    has_atmo: int
    atmo: float               # atmosphere density
    cloud: float              # cloud coverage (0 = clear)
    # animation
    spin: float               # rotation rate


# --------------------------------------------------------------------------- #
# terrain field                                                               #
# --------------------------------------------------------------------------- #

@wp.func
def _fib(k: int, n: int, seed: float) -> wp.vec3:
    y = 1.0 - 2.0 * (float(k) + 0.5) / float(n)
    r = wp.sqrt(wp.max(1.0 - y * y, 0.0))
    th = _GOLDEN * float(k) + seed
    return wp.vec3(r * wp.cos(th), y, r * wp.sin(th))


@wp.func
def _volcano(dir: wp.vec3, cfg: PlanetConfig) -> float:
    """Sum of volcano cones (with a crater dip) along the surface direction."""
    e = float(0.0)
    n = wp.max(cfg.volcano_n, 1)
    for k in range(n):
        vd = _fib(k, n, cfg.seed * 3.1)
        ca = wp.clamp(wp.dot(dir, vd), -1.0, 1.0)
        ang = wp.acos(ca)
        cone = wp.exp(-(ang / 0.20) * (ang / 0.20))
        crater = wp.exp(-(ang / 0.055) * (ang / 0.055))
        e += cone * 0.9 - crater * 0.45
    return e


@wp.func
def _elevation(dir: wp.vec3, cfg: PlanetConfig) -> float:
    """Terrain elevation in relief units (ocean floor negative, peaks positive)."""
    s = wp.vec3(cfg.seed, cfg.seed * 1.7, cfg.seed * 2.3)
    c = domain_warp3(dir * 1.5 + s, 4, 0.5)
    cont = wp.smoothstep(0.46, 0.60, c)                 # 1 land, 0 ocean
    e = (cont - 0.5) * 1.1
    e += (fbm3(dir * 4.0 + s, 5) - 0.5) * 0.5 * cont
    if cfg.mountain > 0.0:
        e += ridged3(dir * 6.0 + s, 5) * cfg.mountain * cont
    if cfg.has_volcano != 0:
        e += _volcano(dir, cfg) * (0.6 + 0.4 * cfg.mountain)
    return e


@wp.func
def _surface_r(dir: wp.vec3, cfg: PlanetConfig) -> float:
    """Rendered surface radius along `dir` — terrain, or water where it floods."""
    e = _elevation(dir, cfg)
    tr = _R * (1.0 + e * _RS)
    if cfg.has_ocean != 0:
        sea = _R * (1.0 + cfg.sea_level * _RS)
        if tr < sea:
            return sea
    return tr


# --------------------------------------------------------------------------- #
# shading                                                                      #
# --------------------------------------------------------------------------- #

@wp.func
def _rs(ro: wp.vec3, rd: wp.vec3, radius: float) -> wp.vec2:
    b = wp.dot(ro, rd)
    c = wp.dot(ro, ro) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return wp.vec2(1.0e30, -1.0e30)
    s = wp.sqrt(disc)
    return wp.vec2(-b - s, -b + s)


@wp.func
def _terrain_normal(dir: wp.vec3, cfg: PlanetConfig) -> wp.vec3:
    # gradient of the surface radius on the sphere, via two tangent offsets
    up = wp.vec3(0.0, 1.0, 0.0)
    if wp.abs(dir[1]) > 0.9:
        up = wp.vec3(1.0, 0.0, 0.0)
    t1 = wp.normalize(wp.cross(up, dir))
    t2 = wp.cross(dir, t1)
    eps = 0.02
    r0 = _surface_r(dir, cfg)
    ra = _surface_r(wp.normalize(dir + t1 * eps), cfg)
    rb = _surface_r(wp.normalize(dir + t2 * eps), cfg)
    # displaced positions, cross for normal
    p0 = dir * r0
    pa = wp.normalize(dir + t1 * eps) * ra
    pb = wp.normalize(dir + t2 * eps) * rb
    n = wp.normalize(wp.cross(pa - p0, pb - p0))
    if wp.dot(n, dir) < 0.0:
        n = -n
    return n


@wp.func
def _volcano_heat(dir: wp.vec3, cfg: PlanetConfig) -> float:
    """How molten the crater is at `dir` (1 at a vent, 0 away from vents)."""
    h = float(0.0)
    n = wp.max(cfg.volcano_n, 1)
    for k in range(n):
        vd = _fib(k, n, cfg.seed * 3.1)
        ang = wp.acos(wp.clamp(wp.dot(dir, vd), -1.0, 1.0))
        h = wp.max(h, wp.exp(-(ang / 0.075) * (ang / 0.075)))
    return h


@wp.func
def _lava_intensity(dir: wp.vec3, e: float, cfg: PlanetConfig) -> float:
    """Molten fraction 0..1 — glowing vents, plus lava seas on a young world."""
    if cfg.lava <= 0.0:
        return 0.0
    crater = float(0.0)
    if cfg.has_volcano != 0:
        crater = _volcano_heat(dir, cfg)
    sea = wp.smoothstep(0.28, -0.30, e) * wp.smoothstep(0.5, 1.0, cfg.lava)
    return wp.clamp(wp.max(crater, sea), 0.0, 1.0)


@wp.func
def _lava_emission(dir: wp.vec3, time: float, cfg: PlanetConfig) -> wp.vec3:
    s = wp.vec3(cfg.seed, cfg.seed * 1.7, cfg.seed * 2.3)
    temp = fbm3(dir * 22.0 + s + wp.vec3(time * 0.08, 0.0, 0.0), 4)
    heat = wp.clamp(temp * 1.4 - 0.2, 0.0, 1.0)
    hot = wp.vec3(0.95, 0.16, 0.02) * (1.0 - heat) + wp.vec3(1.0, 0.72, 0.20) * heat
    # cooled black crust rafts drift across the molten sheet (coarse, coherent)
    crust = wp.smoothstep(0.52, 0.74, domain_warp3(dir * 14.0 + s, 3, 0.8))
    raft = wp.vec3(0.05, 0.02, 0.015)
    return hot * (0.9 + 1.5 * heat) * (1.0 - crust) + raft * crust


@wp.func
def _shade(dir: wp.vec3, n: wp.vec3, rd: wp.vec3, sun: wp.vec3,
           cfg: PlanetConfig, time: float) -> wp.vec3:
    e = _elevation(dir, cfg)
    sea = cfg.sea_level
    s = wp.vec3(cfg.seed, cfg.seed * 1.7, cfg.seed * 2.3)
    v = -rd
    ndl = wp.dot(n, sun)
    day = wp.smoothstep(-0.12, 0.15, wp.dot(dir, sun))
    sun_col = wp.vec3(1.0, 0.97, 0.92)
    lat = wp.abs(dir[1])

    s2 = s + wp.vec3(11.0, 7.0, 3.0)
    s3 = s + wp.vec3(23.0, 17.0, 5.0)
    is_ocean = 0.0
    if cfg.has_ocean != 0 and e < sea:
        is_ocean = 1.0
    # freshwater on land: lakes (isolated basins) + rivers (branching channels)
    fresh = float(0.0)
    if is_ocean < 0.5:
        hh = wp.clamp((e - sea) * 0.7, 0.0, 1.0)
        if cfg.has_lakes != 0:
            fresh = wp.max(fresh, wp.smoothstep(0.34, 0.29, fbm3(dir * 9.0 + s2, 4))
                           * (1.0 - wp.smoothstep(0.4, 0.7, hh)))
        if cfg.has_rivers != 0:
            ed = worley3_f2(dir * 13.0 + s3)
            riv = wp.smoothstep(0.05, 0.005, ed[1] - ed[0])
            fresh = wp.max(fresh, riv * (1.0 - wp.smoothstep(0.55, 0.85, hh)))
    is_water = wp.max(is_ocean, fresh)

    col = wp.vec3(0.0, 0.0, 0.0)
    if is_water > 0.5:
        if is_ocean > 0.5:
            depth = wp.clamp((sea - e) * 2.0, 0.0, 1.0)
            albedo = wp.vec3(0.10, 0.42, 0.48) * (1.0 - depth) \
                + wp.vec3(0.01, 0.05, 0.16) * depth
            lit = shade_pbr(n, v, sun, albedo, 0.06, 0.0, sun_col) * 4.5
        else:
            albedo = wp.vec3(0.04, 0.16, 0.22)       # freshwater — deeper, calmer
            lit = shade_pbr(n, v, sun, albedo, 0.16, 0.0, sun_col) * 2.4
        col = lit + wp.cw_mul(albedo, wp.vec3(0.05, 0.10, 0.16))
    else:
        elev_above = e - sea                             # raw height above sea
        h = wp.clamp(elev_above * 0.6, 0.0, 1.0)          # normalised for colour
        # earthy lowlands -> tan highlands -> bare grey peaks (kept dark, as
        # real land reads from orbit; bloom then stays off the rock)
        low = wp.vec3(0.20, 0.17, 0.11)
        mid = wp.vec3(0.32, 0.26, 0.16)
        peak = wp.vec3(0.30, 0.28, 0.25)
        tex = fbm3(dir * 12.0 + s, 4)
        rock = low * (1.0 - h) + mid * h
        rock = rock * (1.0 - wp.smoothstep(0.55, 0.95, h)) \
            + peak * wp.smoothstep(0.55, 0.95, h)
        rock = rock * (0.74 + 0.28 * tex)                # surface mottling
        rock = rock * (1.0 - 0.55 * cfg.lava)            # dark basalt on molten worlds
        # snow only near the poles or on the very highest peaks (kept separate
        # so raising the land doesn't flood the mid-latitudes with snow)
        polar = wp.smoothstep(0.82, 0.99, lat)
        peak_snow = wp.smoothstep(1.05, 1.6, elev_above)   # only genuine peaks
        snow_m = cfg.snow * wp.clamp(polar + peak_snow, 0.0, 1.0)
        albedo = rock * (1.0 - snow_m) + wp.vec3(0.90, 0.93, 0.98) * snow_m
        lit = shade_pbr(n, v, sun, albedo, 0.88, 0.0, sun_col) * 1.15
        col = albedo * (0.04 + 0.72 * wp.max(ndl, 0.0)) * 0.5 + lit

    # molten lava: emissive, overrides surface lighting where hot
    lava_i = _lava_intensity(dir, e, cfg)
    if lava_i > 0.01:
        emit = _lava_emission(dir, time, cfg)
        col = col * (1.0 - lava_i) + emit * lava_i

    return col


# --------------------------------------------------------------------------- #
# atmosphere (this module's own, normalised-scale)                             #
# --------------------------------------------------------------------------- #

@wp.func
def _atmosphere(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3, cfg: PlanetConfig,
                surf_t: float) -> wp.vec3:
    if cfg.has_atmo == 0:
        return wp.vec3(0.0, 0.0, 0.0)
    ra = _R * 1.06
    hit = _rs(ro, rd, ra)
    if hit[1] < 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    t0 = wp.max(hit[0], 0.0)
    t1 = hit[1]
    if surf_t > 0.0:
        t1 = wp.min(t1, surf_t)
    if t1 <= t0:
        return wp.vec3(0.0, 0.0, 0.0)
    # cheap single-scatter: integrate density along the segment, sun-tinted
    scatter = wp.vec3(0.0, 0.0, 0.0)
    steps = 10
    seg = (t1 - t0) / float(steps)
    t = t0 + 0.5 * seg
    rayleigh = wp.vec3(0.24, 0.44, 1.0)
    for _ in range(steps):
        p = ro + rd * t
        h = (wp.length(p) - _R) / (ra - _R)
        dens = wp.exp(-wp.clamp(h, 0.0, 1.0) * 3.0) * seg
        sun_up = wp.clamp(wp.dot(wp.normalize(p), sun) + 0.25, 0.0, 1.0)
        scatter += rayleigh * (dens * sun_up)
        t += seg
    phase = 0.75 * (1.0 + wp.dot(rd, sun) * wp.dot(rd, sun))
    return wp.cw_mul(scatter, rayleigh) * (cfg.atmo * 2.2 * phase)


# --------------------------------------------------------------------------- #
# kernel                                                                        #
# --------------------------------------------------------------------------- #

@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  cfg: PlanetConfig, time: float, steps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    # spin the planet by rotating the ray about Y (cheap "planet turns")
    a = -time * cfg.spin
    ca = wp.cos(a)
    sa = wp.sin(a)
    ro = wp.vec3(ca * ro[0] - sa * ro[2], ro[1], sa * ro[0] + ca * ro[2])
    rd = wp.vec3(ca * rd[0] - sa * rd[2], rd[1], sa * rd[0] + ca * rd[2])
    sr = wp.vec3(ca * sun[0] - sa * sun[2], sun[1], sa * sun[0] + ca * sun[2])

    col = stars(rd)

    rmax = _R * (1.0 + 1.2 * _RS)
    outer = _rs(ro, rd, rmax)
    surf_t = float(-1.0)
    if outer[1] > 0.0:
        t = wp.max(outer[0], 0.0)
        t1 = outer[1]
        dt = (t1 - t) / float(steps)
        prev = float(1.0)                       # sign of (r - surface_r)
        hit_t = float(-1.0)
        for _ in range(steps):
            p = ro + rd * t
            r = wp.length(p)
            d = p / r
            srad = _surface_r(d, cfg)
            f = r - srad
            if f < 0.0:
                hit_t = t
                # one bisection refinement
                ta = t - dt
                tb = t
                for _b in range(6):
                    tm = 0.5 * (ta + tb)
                    pm = ro + rd * tm
                    rm = wp.length(pm)
                    if rm - _surface_r(pm / rm, cfg) < 0.0:
                        tb = tm
                    else:
                        ta = tm
                hit_t = tb
                break
            t += dt
        if hit_t > 0.0:
            p = ro + rd * hit_t
            d = wp.normalize(p)
            n = _terrain_normal(d, cfg)
            col = _shade(d, n, rd, sr, cfg, time)
            surf_t = hit_t

    col = col + _atmosphere(ro, rd, sr, cfg, surf_t)
    img[i, j] = col


def make_config(**kw) -> PlanetConfig:
    """Build a PlanetConfig with sensible defaults; override any field by name."""
    cfg = PlanetConfig()
    cfg.seed = float(kw.get("seed", 1.0))
    cfg.mountain = float(kw.get("mountain", 0.6))
    cfg.sea_level = float(kw.get("sea_level", 0.0))
    cfg.has_ocean = int(kw.get("has_ocean", 1))
    cfg.has_lakes = int(kw.get("has_lakes", 0))
    cfg.has_rivers = int(kw.get("has_rivers", 0))
    cfg.snow = float(kw.get("snow", 1.0))
    cfg.has_volcano = int(kw.get("has_volcano", 0))
    cfg.volcano_n = int(kw.get("volcano_n", 4))
    cfg.lava = float(kw.get("lava", 0.0))
    cfg.veg = float(kw.get("veg", 0.0))
    cfg.alive = float(kw.get("alive", 0.0))
    cfg.city = float(kw.get("city", 0.0))
    cfg.has_atmo = int(kw.get("has_atmo", 1))
    cfg.atmo = float(kw.get("atmo", 1.0))
    cfg.cloud = float(kw.get("cloud", 0.0))
    cfg.spin = float(kw.get("spin", 0.05))
    return cfg


def _steps_for(quality: str) -> int:
    return {"low": 64, "medium": 110, "high": 170, "ultra": 260}.get(quality, 110)


def render_planet(cfg: PlanetConfig, width: int, height: int, time: float = 0.0,
                  mouse=(0.0, 0.0), device: str = "cpu", quality: str = "medium",
                  sun_az: float = 1.1, sun_el: float = 0.35,
                  dist: float = 3.4) -> np.ndarray:
    """Render `cfg` to an ``(H, W, 3)`` image."""
    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.28 + float(mouse[1]) * 0.003
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=42.0, aspect=width / height)
    sun = wp.vec3(math.cos(sun_el) * math.sin(sun_az), math.sin(sun_el),
                  math.cos(sun_el) * math.cos(sun_az))
    steps = _steps_for(quality)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, cfg, float(time), int(steps),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=2.4, strength=0.45, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.02)
