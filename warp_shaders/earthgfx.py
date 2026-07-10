"""Shared realistic-Earth shading — used by both the `earth` scene and the
Earth blast simulation (so the simulated globe looks identical to the scene).

``earth_color(ro, rd, sun, time, soot)`` is the whole shader: starfield +
ray-sphere globe (procedural continents, oceans w/ sun-glint, clouds, day/night
terminator + city lights) + atmospheric-scattering rim. ``soot`` in [0,1] greys
and dims the surface for the toxic / nuclear-winter outcome.
"""

import warp as wp

from .particles import noise3, ray_sphere


@wp.func
def fbm3(p: wp.vec3) -> float:
    v = float(0.0)
    a = float(0.5)
    for _ in range(5):
        v += a * noise3(p)
        p = p * 2.03 + wp.vec3(1.7, 9.2, 3.3)
        a *= 0.5
    return v


@wp.func
def mix3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:
    return wp.vec3(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


@wp.func
def sstep(e0: float, e1: float, x: float) -> float:
    t = wp.clamp((x - e0) / (e1 - e0 + 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def stars(rd: wp.vec3) -> wp.vec3:
    s = wp.pow(noise3(rd * 220.0), 34.0) * 3.0
    tint = 0.7 + 0.3 * noise3(rd * 50.0 + wp.vec3(4.0, 4.0, 4.0))
    return wp.vec3(0.8, 0.85, 1.0) * (s * tint)


@wp.func
def surface(n: wp.vec3, rd: wp.vec3, sun: wp.vec3, time: float, soot: float) -> wp.vec3:
    warp = fbm3(n * 1.4)
    c = fbm3(n * 2.3 + wp.vec3(warp, warp, warp) * 0.7)
    detail = fbm3(n * 6.0)
    land = sstep(0.57, 0.62, c + 0.06 * detail)

    lat = wp.abs(n[1])
    elev = fbm3(n * 4.0)
    dry = fbm3(n * 3.0 + wp.vec3(5.0, 5.0, 5.0))

    ocean = mix3(wp.vec3(0.015, 0.06, 0.20), wp.vec3(0.04, 0.22, 0.38), sstep(0.35, 0.52, c))
    land_c = mix3(wp.vec3(0.10, 0.28, 0.07), wp.vec3(0.52, 0.42, 0.24), sstep(0.42, 0.72, dry))
    snow = sstep(0.62, 0.78, elev * 0.5 + sstep(0.7, 0.92, lat))
    land_c = mix3(land_c, wp.vec3(0.90, 0.92, 0.96), snow)
    base = mix3(ocean, land_c, land)
    is_ocean = 1.0 - land

    ndl = wp.dot(n, sun)
    day = sstep(-0.12, 0.18, ndl)
    diff = wp.max(ndl, 0.0)
    col = base * (0.05 + 0.95 * diff)

    view = -rd
    h = wp.normalize(sun + view)
    spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 60.0) * day * is_ocean
    col = col + wp.vec3(1.0, 0.95, 0.8) * (spec * 0.9)

    cl = sstep(0.60, 0.76, fbm3(n * 2.6 + wp.vec3(time * 0.02, 0.0, time * 0.01)))
    col = mix3(col, wp.vec3(1.0, 1.0, 1.0) * (0.08 + 0.92 * diff), cl * 0.82)

    night = 1.0 - day
    city = land * sstep(0.56, 0.70, fbm3(n * 9.0)) * (1.0 - cl)
    col = col + wp.vec3(1.0, 0.72, 0.32) * (city * night * 2.6)

    fres = wp.pow(1.0 - wp.max(wp.dot(n, view), 0.0), 3.0)
    term = sstep(0.0, 0.35, ndl) * (1.0 - sstep(0.35, 0.7, ndl))
    col = col + wp.vec3(0.35, 0.55, 1.0) * (fres * day * 1.5)
    col = col + wp.vec3(1.0, 0.45, 0.2) * (fres * term * 0.6)

    # Nuclear-winter soot: grey the surface + dim it.
    if soot > 0.0:
        grey = wp.vec3(0.16, 0.15, 0.14)
        col = mix3(col, grey * (0.1 + 0.9 * diff), soot * 0.85) * (1.0 - 0.5 * soot)
    return col


@wp.func
def earth_color(ro: wp.vec3, rd: wp.vec3, sun: wp.vec3, time: float, soot: float) -> wp.vec3:
    """Full realistic-Earth color for a view ray. Linear (no gamma)."""
    center = wp.vec3(0.0, 0.0, 0.0)
    te0, te1, ehit = ray_sphere(ro, rd, center, 1.0)
    ta0, ta1, ahit = ray_sphere(ro, rd, center, 1.06)

    col = stars(rd)
    if ehit == 1 and te0 > 0.0:
        p = ro + rd * te0
        col = surface(wp.normalize(p), rd, sun, time, soot)
    elif ahit == 1:
        t0 = wp.max(ta0, 0.0)
        seg = ta1 - t0
        nn = wp.normalize(ro + rd * (0.5 * (t0 + ta1)))
        lit = sstep(-0.25, 0.35, wp.dot(nn, sun))
        forward = wp.pow(wp.max(wp.dot(rd, sun), 0.0), 4.0)
        acol = mix3(wp.vec3(0.30, 0.5, 1.0), wp.vec3(1.0, 0.5, 0.2), forward)
        glow = wp.min(seg * 3.2, 1.6) * lit
        col = col * (1.0 - wp.min(glow, 1.0)) + acol * glow
    return col
