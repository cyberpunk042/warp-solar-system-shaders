"""Ringed exotic planet — a crystalline ice world girdled by a bright ring system.

Adapts the gas-giant ring machinery (ray-plane annulus + radial bands + Cassini
gap + mutual planet/ring shadowing) to an **exotic ice world**: a blue-white,
crevassed, subsurface-glowing body under an icy, brighter ring, with a small moon
in attendance. See ``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, ridged3, value3
from ..scene import Scene

_R = 1.0
_RING_IN = 1.4
_RING_OUT = 2.6
_MOON = wp.constant(wp.vec3(3.6, 0.7, -1.4))     # moon centre
_MOON_R = 0.28


@wp.func
def _ring_density(r: float) -> float:
    if r < _RING_IN or r > _RING_OUT:
        return 0.0
    x = (r - _RING_IN) / (_RING_OUT - _RING_IN)
    bands = 0.5 + 0.3 * wp.sin(r * 60.0) + 0.28 * wp.sin(r * 24.0) + 0.15 * wp.sin(r * 9.0)
    cassini = wp.smoothstep(0.48, 0.52, x) * (1.0 - wp.smoothstep(0.56, 0.6, x))
    dens = bands * (1.0 - 0.92 * cassini)
    edge = wp.smoothstep(0.0, 0.04, x) * (1.0 - wp.smoothstep(0.92, 1.0, x))
    return wp.clamp(dens * edge, 0.0, 1.0)


@wp.func
def _ice_color(n: wp.vec3, rd: wp.vec3, sun: wp.vec3, ro: wp.vec3) -> wp.vec3:
    # crystalline ice: crevasses + ridges, swirled cool bands
    crack = ridged3(n * 6.0, 5)
    swirl = fbm3(n * 2.4 + wp.vec3(crack, crack, 0.0) * 0.5, 4)
    ice = wp.vec3(0.62, 0.78, 0.98)                       # blue-white
    deep = wp.vec3(0.16, 0.34, 0.60)                      # deep crevasse blue
    frost = wp.vec3(0.92, 0.96, 1.0)
    col = deep * (1.0 - crack) + ice * crack
    col = col * (1.0 - wp.smoothstep(0.6, 0.9, swirl)) + frost * wp.smoothstep(0.6, 0.9, swirl)
    ndl = wp.max(wp.dot(n, sun), 0.0)
    limb = wp.pow(wp.max(wp.dot(n, -rd), 0.0), 0.5)
    # subsurface rim glow (cyan) where the limb is backlit
    rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
    sub = wp.vec3(0.25, 0.55, 0.85) * (rim * (0.3 + 0.7 * ndl))
    # ring shadow on the planet
    p = ro + rd * _rs(ro, rd, _R)[0]
    shadow = float(1.0)
    if wp.abs(sun[1]) > 0.001:
        ts = -p[1] / (sun[1] + 1e-6)
        if ts > 0.0:
            cr = wp.length(wp.vec2(p[0] + sun[0] * ts, p[2] + sun[2] * ts))
            shadow = 1.0 - 0.85 * _ring_density(cr)
    return col * (0.06 + 0.95 * ndl * shadow) * limb + sub


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    st = wp.pow(value3(rd * 300.0), 40.0) * 4.0
    col = wp.vec3(0.7, 0.8, 1.0) * st + wp.vec3(0.008, 0.010, 0.018)   # dark space
    g = _rs(ro, rd, _R)
    planet_hit = g[0] > 0.0 and g[0] < 1.0e29             # (miss sentinel is +1e30)
    p_col = wp.vec3(0.0, 0.0, 0.0)
    if planet_hit:
        n = wp.normalize(ro + rd * g[0])
        p_col = _ice_color(n, rd, sun, ro)

    # ring intersection (plane y = 0)
    ring_hit = int(0)
    t_r = float(1e30)
    r_col = wp.vec3(0.0, 0.0, 0.0)
    r_a = float(0.0)
    if wp.abs(rd[1]) > 1e-5:
        t_r = -ro[1] / rd[1]
        if t_r > 0.0:
            pr = ro + rd * t_r
            rr = wp.length(wp.vec2(pr[0], pr[2]))
            dens = _ring_density(rr)
            if dens > 0.001:
                ring_hit = 1
                shade = wp.max(wp.dot(wp.vec3(0.0, 1.0, 0.0), sun), 0.15) + 0.25
                ps = _rs(pr, sun, _R)
                if ps[0] > 0.0 and ps[1] > 0.0:
                    shade = shade * 0.25                  # planet shadow on rings
                base = wp.vec3(0.72, 0.82, 1.0) * (0.55 + 0.45 * dens)   # icy rings
                r_col = base * shade
                r_a = wp.clamp(dens * 1.15, 0.0, 1.0)

    # composite planet + rings
    if planet_hit == True and ring_hit == 1:
        if g[0] < t_r:
            col = p_col
        else:
            col = p_col * (1.0 - r_a) + r_col * r_a
    elif ring_hit == 1:
        col = col * (1.0 - r_a) + r_col * r_a
    elif planet_hit == True:
        col = p_col

    # the moon (a small cratered body) — composited if it is the nearest hit
    gm = _rs(ro - _MOON, rd, _MOON_R)
    if gm[0] > 0.0 and gm[0] < 1.0e29:
        p_depth = g[0]
        if p_depth <= 0.0 or p_depth > 1.0e29:
            p_depth = 1.0e9                               # planet missed
        r_depth = t_r
        if ring_hit == 0:
            r_depth = 1.0e9
        if gm[0] < p_depth and gm[0] < r_depth:
            nm = wp.normalize((ro - _MOON) + rd * gm[0])
            crat = fbm3(nm * 8.0, 4)
            mcol = wp.vec3(0.5, 0.52, 0.58) * (0.7 + 0.4 * crat)
            ndlm = wp.max(wp.dot(nm, sun), 0.0)
            col = mcol * (0.05 + 0.95 * ndlm)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.7 + float(mouse[0]) * 0.01 + time * 0.04
    el = 0.5
    dist = 6.6
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=36.0, aspect=width / height)

    sel = 0.2 + float(mouse[1]) * 0.004
    saz = az + 1.2
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), math.cos(sel) * math.cos(saz))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.25, strength=0.35, radius=max(2, int(width * 0.01)), passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="ringed_planet",
    description="An exotic crystalline ice world girdled by a bright icy ring "
                "system (mutual planet/ring shadowing) with an attendant moon.",
    renderer=_render,
)
