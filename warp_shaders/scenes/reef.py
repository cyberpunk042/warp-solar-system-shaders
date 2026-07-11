"""Underwater — a sandy seabed under rippling caustics and surface god-rays.

The camera sits below the surface looking across a gently rippled sand seabed.
Moving **caustics** (Worley F2−F1 network bands, two animated octaves) focus
bright light on the floor; blue-green **depth extinction** (`apply_fog`) swallows
distance into teal; and downward **god-ray** shafts fall from the bright surface
via `post.godrays`. Heightfield raymarch (crossing-detection + bisection);
`--quality` scales march/shadow steps. iMouse pans.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm_perlin3, worley3_f2
from ..scene import Scene

_FAR = 220.0
_WATER = wp.constant(wp.vec3(0.06, 0.28, 0.34))     # deep-water tint


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.03, 0.0, z * 0.03)
    dunes = fbm_perlin3(p, 5) * 2.6
    ripple = wp.sin(x * 0.6 + z * 0.2) * 0.18 + wp.sin(z * 0.9 - x * 0.3) * 0.12
    return dunes + ripple - 1.0


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.05
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _caustics(x: float, z: float, time: float) -> float:
    # bright network where Voronoi cells meet, two drifting octaves
    a = worley3_f2(wp.vec3(x * 0.22 + time * 0.35, time * 0.1, z * 0.22))
    b = worley3_f2(wp.vec3(x * 0.5 - time * 0.25, time * 0.15, z * 0.5 + 3.0))
    ca = wp.pow(wp.clamp(1.0 - (a[1] - a[0]) * 3.2, 0.0, 1.0), 2.0)
    cb = wp.pow(wp.clamp(1.0 - (b[1] - b[0]) * 3.6, 0.0, 1.0), 2.0)
    return ca * 0.7 + cb * 0.4


@wp.func
def _watercol(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = _WATER * (1.0 - up) + wp.vec3(0.12, 0.5, 0.6) * up   # brighter toward surface
    return base + sun_disk(rd, sun, wp.vec3(0.7, 0.95, 1.0), 0.9985, 0.8) * up


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.3)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2])
        if h < 0.001:
            return 0.0
        res = wp.min(res, 9.0 * h / t)
        t += wp.clamp(h, 0.3, 5.0)
        if t > 70.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  time: float, march_steps: int, shadow_steps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.3)
    prev_t = t
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = p[1] - _height(p[0], p[2])
        if d < 0.0:
            hit = 1
            break
        prev_t = t
        t += wp.max(d * 0.45, 0.008 * t)
        if t > _FAR:
            break

    if hit == 0:
        img[i, j] = _watercol(rd, sun)
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

    sand = wp.vec3(0.66, 0.62, 0.46)
    sh = _shadow(p + n * 0.04, sun, shadow_steps)
    direct = shade_pbr(n, v_dir, sun, sand, 0.7, 0.0, wp.vec3(1.0, 0.97, 0.85)) * (2.4 * sh)
    amb = wp.cw_mul(wp.vec3(0.16, 0.4, 0.46), sand) * (0.5 + 0.5 * n[1])
    # caustics: focused light dancing on the lit floor (needs sun visibility)
    caus = _caustics(p[0], p[2], time) * sh * (0.4 + 0.6 * n[1])
    col = direct + amb + wp.vec3(0.7, 0.95, 1.0) * (caus * 1.3)

    # blue-green depth extinction: distance drowns everything into teal
    col = apply_fog(col, t, _WATER, 0.018)
    img[i, j] = col


def _counts(name):
    return {"low": (120, 12), "medium": (200, 18), "high": (300, 26),
            "ultra": (440, 40)}.get(name, (200, 18))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.4 + float(mouse[0]) * 0.006 + time * 0.03
    pitch = -0.16 + float(mouse[1]) * 0.004         # look slightly down at the seabed
    eye = (0.0, 9.0, 0.0)
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=68.0, aspect=width / height)

    # sun near the surface, ahead + up, so god-rays fall into frame from above
    el = 0.62
    sun_np = np.array([math.sin(az) * math.cos(el), math.sin(el),
                       math.cos(az) * math.cos(el)], np.float32)
    sun = wp.vec3(float(sun_np[0]), float(sun_np[1]), float(sun_np[2]))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time), int(ms), int(ss),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(2, int(min(width, height) * 0.014))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.45, radius=r, passes=2)

    # god-ray shafts from the surface sun
    fwd_n = np.array(fwd, np.float32)
    fwd_n /= np.linalg.norm(fwd_n) + 1e-9
    right = np.cross(fwd_n, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd_n)
    cz = float(sun_np @ fwd_n)
    if cz > 0.05:
        thf = math.tan(math.radians(68.0) * 0.5)
        asp = width / height
        cx = 0.5 + 0.5 * (float(sun_np @ right) / cz) / (asp * thf)
        cy = 0.5 - 0.5 * (float(sun_np @ upv) / cz) / thf
        hdr = post.godrays(hdr, cx, cy, samples=44, density=0.94, weight=0.7, threshold=1.0)

    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="reef",
    description="Underwater seabed: rippling caustics, blue-green depth, surface god-rays. --quality low..ultra.",
    renderer=_render,
)
