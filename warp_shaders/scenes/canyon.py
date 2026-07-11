"""Slot canyon — layered sandstone walls and volumetric god-rays.

A heightfield with a narrow, winding slot carved into a tall sandstone block:
the camera stands on the slot floor looking up its length toward a bright sky
gap, so the walls frame the sun and `post.godrays` throws dusty light shafts
down between them. Walls are warm PBR sandstone with height-banded strata; soft
shadows deepen the slot; warm haze fills the depths. The sun sits on-screen in
the gap to drive the god-rays. Heightfield raymarch (crossing-detection +
bisection); `--quality` scales march/shadow steps. iMouse pans.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm_perlin3
from ..scene import Scene

_FAR = 240.0
_WALL = 22.0          # wall top height


@wp.func
def _center(z: float) -> float:
    # the slot meanders in x as it runs along z
    return wp.sin(z * 0.14) * 4.5 + wp.sin(z * 0.37 + 1.3) * 1.6


@wp.func
def _half_w(z: float) -> float:
    return 2.1 + 0.7 * wp.sin(z * 0.21 + 0.5)


@wp.func
def _height(x: float, z: float) -> float:
    dx = wp.abs(x - _center(z))
    hw = _half_w(z)
    # 0 on the slot floor, rising to the wall top just outside the slot mouth
    wall = wp.smoothstep(hw, hw + 3.4, dx)
    p = wp.vec3(x * 0.05, 0.0, z * 0.05)
    rough = fbm_perlin3(p, 5) * 2.2 + fbm_perlin3(p * 3.3, 4) * 0.7
    return wall * _WALL + wall * rough


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.04
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.55, 0.66, 0.86) * up + wp.vec3(0.9, 0.8, 0.62) * (1.0 - up)
    return base + sun_disk(rd, sun, wp.vec3(1.0, 0.92, 0.75), 0.9989, 0.7)


@wp.func
def _strata(y: float) -> wp.vec3:
    # sedimentary colour banding by height
    b = 0.5 + 0.5 * wp.sin(y * 1.7)
    b2 = 0.5 + 0.5 * wp.sin(y * 0.6 + 1.0)
    dark = wp.vec3(0.42, 0.20, 0.12)
    mid = wp.vec3(0.72, 0.40, 0.22)
    pale = wp.vec3(0.85, 0.62, 0.42)
    return mid * (1.0 - b) + pale * b * b2 + dark * (1.0 - b2) * 0.4


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.3)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2])
        if h < 0.001:
            return 0.0
        res = wp.min(res, 10.0 * h / t)
        t += wp.clamp(h, 0.25, 4.0)
        if t > 60.0:
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

    t = float(0.2)
    prev_t = t
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = p[1] - _height(p[0], p[2])
        if d < 0.0:
            hit = 1
            break
        prev_t = t
        t += wp.max(d * 0.4, 0.006 * t)
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

    albedo = _strata(p[1])
    sh = _shadow(p + n * 0.03, sun, shadow_steps)
    direct = shade_pbr(n, v_dir, sun, albedo, 0.75, 0.0, wp.vec3(1.0, 0.82, 0.55)) * (3.2 * sh)
    # warm bounce fill from the sunlit walls so the shaded slot isn't black
    amb = wp.cw_mul(wp.vec3(0.5, 0.42, 0.38), albedo) * (0.5 + 0.5 * n[1])
    col = direct + amb

    col = apply_fog(col, t, wp.vec3(0.5, 0.36, 0.26), 0.006)
    img[i, j] = col


def _counts(name):
    return {"low": (130, 12), "medium": (220, 18), "high": (340, 26),
            "ultra": (500, 40)}.get(name, (220, 18))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    # walk down the slot; keep the eye on the winding floor
    zc = time * 1.5
    xc = math.sin(zc * 0.14) * 4.5 + math.sin(zc * 0.37 + 1.3) * 1.6
    az = float(mouse[0]) * 0.006
    pitch = 0.42 + float(mouse[1]) * 0.004        # look up toward the sky gap
    eye = (xc, 2.6, zc)
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=66.0, aspect=width / height)

    # sun in the gap, down the view axis so it frames on-screen and lights shafts
    el = 0.92
    sun_np = np.array([math.sin(az) * math.cos(el), math.sin(el),
                       math.cos(az) * math.cos(el)], np.float32)
    sun = wp.vec3(float(sun_np[0]), float(sun_np[1]), float(sun_np[2]))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.4, radius=r, passes=2)

    # project the sun to screen and throw god-ray shafts down the slot
    fwd_n = np.array(fwd, np.float32)
    fwd_n /= np.linalg.norm(fwd_n) + 1e-9
    right = np.cross(fwd_n, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd_n)
    cz = float(sun_np @ fwd_n)
    if cz > 0.05:
        thf = math.tan(math.radians(66.0) * 0.5)
        asp = width / height
        cx = 0.5 + 0.5 * (float(sun_np @ right) / cz) / (asp * thf)
        cy = 0.5 - 0.5 * (float(sun_np @ upv) / cz) / thf
        hdr = post.godrays(hdr, cx, cy, samples=48, density=0.95, weight=0.85, threshold=1.2)

    return post.tonemap(hdr, mode="aces", exposure=1.02)


SCENE = Scene(
    name="canyon",
    description="Slot canyon: layered sandstone walls + volumetric god-rays. --quality low..ultra.",
    renderer=_render,
)
