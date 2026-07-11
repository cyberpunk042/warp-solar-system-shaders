"""Gas giant with rings — a Saturn/Jupiter-like world.

Banded atmosphere (domain-warped latitude bands + a great red spot), a flat
ring system (ray-plane annulus with radial band structure and a Cassini-style
gap), and the mutual shadowing that sells it: the rings cast a shadow band on the
planet, and the planet casts its shadow across the rings. Stars behind.
`--quality` unused here (analytic); iMouse orbits / moves the sun.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from ..scene import Scene

_R = 1.0
_RING_IN = 1.35
_RING_OUT = 2.35


@wp.func
def _ring_density(r: float) -> float:
    if r < _RING_IN or r > _RING_OUT:
        return 0.0
    x = (r - _RING_IN) / (_RING_OUT - _RING_IN)          # 0..1 across the rings
    bands = 0.5 + 0.5 * wp.sin(r * 55.0) * 0.6 + 0.3 * wp.sin(r * 21.0)
    cassini = wp.smoothstep(0.42, 0.46, x) * (1.0 - wp.smoothstep(0.5, 0.54, x))
    dens = bands * (1.0 - 0.9 * cassini)
    edge = wp.smoothstep(0.0, 0.05, x) * (1.0 - wp.smoothstep(0.9, 1.0, x))
    return wp.clamp(dens * edge, 0.0, 1.0)


@wp.func
def _planet_color(n: wp.vec3, rd: wp.vec3, sun: wp.vec3, ro: wp.vec3) -> wp.vec3:
    warp = fbm3(n * 2.2, 4) * 0.18
    y = n[1] + warp
    band = fbm3(wp.vec3(0.0, y * 8.0, 0.0) + n * 0.25, 3)
    cream = wp.vec3(0.85, 0.78, 0.62)
    tan = wp.vec3(0.70, 0.52, 0.34)
    brown = wp.vec3(0.45, 0.30, 0.20)
    white = wp.vec3(0.92, 0.90, 0.85)
    col = tan * (1.0 - band) + cream * band
    col = col * (1.0 - wp.smoothstep(0.6, 0.85, band)) + white * wp.smoothstep(0.6, 0.85, band)
    col = col * (1.0 - wp.smoothstep(0.15, 0.0, band)) + brown * wp.smoothstep(0.15, 0.0, band)

    # great red spot
    spot = wp.vec3(0.55, -0.35, 0.75)
    dsp = wp.length(n - wp.normalize(spot))
    red = wp.smoothstep(0.34, 0.16, dsp)
    col = col * (1.0 - red) + wp.vec3(0.72, 0.28, 0.16) * red

    ndl = wp.max(wp.dot(n, sun), 0.0)
    limb = wp.pow(wp.max(wp.dot(n, -rd), 0.0), 0.4)

    # ring shadow on the planet: does the sun ray cross the ring annulus?
    p = ro + rd * _rs(ro, rd, _R)[0]
    shadow = float(1.0)
    if sun[1] * p[1] < 0.0 or wp.abs(sun[1]) > 0.001:
        ts = -p[1] / (sun[1] + 1e-6)
        if ts > 0.0:
            cr = wp.length(wp.vec2(p[0] + sun[0] * ts, p[2] + sun[2] * ts))
            shadow = 1.0 - 0.8 * _ring_density(cr)
    return col * (0.05 + 0.95 * ndl * shadow) * limb


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = stars(rd)
    g = _rs(ro, rd, _R)
    planet_hit = g[0] > 0.0
    p_col = wp.vec3(0.0, 0.0, 0.0)
    if planet_hit:
        n = wp.normalize(ro + rd * g[0])
        p_col = _planet_color(n, rd, sun, ro)

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
                shade = wp.max(wp.dot(wp.vec3(0.0, 1.0, 0.0), sun), 0.15) + 0.2
                # planet shadow on the rings
                ps = _rs(pr, sun, _R)
                if ps[0] > 0.0 and ps[1] > 0.0:
                    shade = shade * 0.25
                base = wp.vec3(0.78, 0.72, 0.6) * (0.6 + 0.4 * dens)
                r_col = base * shade
                r_a = wp.clamp(dens * 1.1, 0.0, 1.0)

    if planet_hit == True and ring_hit == 1:
        if g[0] < t_r:
            col = p_col                                   # planet in front
        else:
            col = p_col * (1.0 - r_a) + r_col * r_a        # ring in front of planet
    elif ring_hit == 1:
        col = col * (1.0 - r_a) + r_col * r_a
    elif planet_hit == True:
        col = p_col
    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.7 + float(mouse[0]) * 0.01 + time * 0.04
    el = 0.42
    dist = 6.2
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=34.0, aspect=width / height)

    sel = 0.15 + float(mouse[1]) * 0.004
    saz = az + 1.1
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), math.cos(sel) * math.cos(saz))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.3, strength=0.3, radius=max(2, int(width * 0.01)), passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="gas_giant",
    description="Ringed gas giant: banded atmosphere, great red spot, rings with mutual shadowing.",
    renderer=_render,
)
