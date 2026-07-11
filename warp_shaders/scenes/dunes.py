"""Desert dunes — a wind-sculpted sand sea at low sun.

A heightfield of meandering dune crests (warped ridge waves at two scales) with
fine wind ripples layered on top, shaded as warm PBR sand under a low sun for
long raking shadows and a bright crest rim. Aerial-perspective haze fades the
dunes into a warm sky. Heightfield raymarch with crossing-detection + bisection;
`--quality` scales march/shadow steps. iMouse pans.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sky_gradient, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm_perlin3
from ..scene import Scene

_FAR = 340.0


@wp.func
def _ridge(a: float) -> float:
    # rounded, slightly asymmetric dune crest from a phase `a`
    s = 0.5 + 0.5 * wp.sin(a)
    return wp.pow(s, 1.4)


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.012, 0.0, z * 0.012)
    # meander the dune lines so crests wander instead of running dead straight
    m = fbm_perlin3(p * 0.6, 4) * 2.2
    big = _ridge(x * 0.045 + z * 0.012 + m) * 9.0
    med = _ridge(x * 0.02 - z * 0.11 + m * 0.7) * 4.0
    # fine wind ripples across the windward faces
    ripple = wp.sin(x * 0.7 + z * 0.22 + m * 3.0) * 0.28
    ripple += wp.sin(x * 1.7 - z * 0.5) * 0.12
    return big + med + ripple - 2.0


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.05
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    base = sky_gradient(rd, wp.vec3(0.85, 0.72, 0.55), wp.vec3(0.28, 0.42, 0.70))
    return base + sun_disk(rd, sun, wp.vec3(1.0, 0.85, 0.6), 0.9990, 0.6)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.4)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2])
        if h < 0.001:
            return 0.0
        res = wp.min(res, 14.0 * h / t)
        t += wp.clamp(h, 0.35, 6.0)
        if t > 130.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  march_steps: int, shadow_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(1.0)
    prev_t = t
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = p[1] - _height(p[0], p[2])
        if d < 0.0:
            hit = 1
            break
        prev_t = t
        t += wp.max(d * 0.5, 0.01 * t)
        if t > _FAR:
            break

    if hit == 0:
        img[i, j] = _sky(rd, sun)
        return

    a = prev_t
    b = t
    for _ in range(6):
        m = 0.5 * (a + b)
        pm = ro + rd * m
        if pm[1] - _height(pm[0], pm[2]) < 0.0:
            b = m
        else:
            a = m
    t = 0.5 * (a + b)
    p = ro + rd * t
    n = _normal(p[0], p[2])
    v_dir = -rd

    sh = _shadow(p + n * 0.04, sun, shadow_steps)
    sand = wp.vec3(0.82, 0.52, 0.24)
    # paler, warmer sand on the crests; deeper ochre in the troughs
    crest = wp.smoothstep(4.0, 10.0, p[1])
    sand = sand * (1.0 - crest) + wp.vec3(0.9, 0.66, 0.38) * crest
    direct = shade_pbr(n, v_dir, sun, sand, 0.6, 0.0, wp.vec3(1.0, 0.8, 0.52)) * (3.4 * sh)
    # warm skylight ambient (not the blue zenith) so shadows stay desert-warm
    amb = wp.cw_mul(wp.vec3(0.55, 0.5, 0.55), sand) * (0.28 * (0.5 + 0.5 * n[1]))
    col = direct + amb

    col = apply_fog(col, t, wp.vec3(0.88, 0.72, 0.5), 0.0032)
    img[i, j] = col


def _counts(name):
    return {"low": (110, 16), "medium": (180, 24), "high": (280, 32),
            "ultra": (420, 48)}.get(name, (180, 24))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.4 + float(mouse[0]) * 0.008 + time * 0.02
    eye = (math.sin(az) * 5.0, 8.0, math.cos(az) * 5.0)
    target = (eye[0] + math.sin(az) * 10.0, 5.0 + float(mouse[1]) * 0.02,
              eye[2] + math.cos(az) * 10.0)
    cam = make_camera(eye, target, fov_deg=64.0, aspect=width / height)

    el = 0.12                         # low sun -> long shadows
    sun = wp.vec3(math.sin(az + 2.2) * math.cos(el), math.sin(el),
                  math.cos(az + 2.2) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.5, strength=0.25, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=0.95)


SCENE = Scene(
    name="dunes",
    description="Wind-sculpted desert dunes at low sun (PBR sand, long shadows, aerial haze). --quality low..ultra.",
    renderer=_render,
)
