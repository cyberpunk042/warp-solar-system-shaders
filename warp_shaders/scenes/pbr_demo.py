"""PBR raymarch demo — the P2 engine showcase (and the copy-me raymarch template).

Sphere-traces an SDF scene (a ground plane + three spheres of increasing
roughness, one metallic) at the active LOD tier, with gradient normals, IQ soft
shadows, ambient occlusion, GGX PBR shading, a procedural sky, and the host post
pipeline (bloom + ACES + vignette). Camera/Light/Quality/Frame arrive as the
engine's @wp.struct uniforms. Try `--quality low|medium|high|ultra`.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.material import make_mat, shade_material
from ..engine.uniforms import (
    Camera, Frame, Light, Quality, camera_ray_dir, make_camera, make_frame,
    make_light, make_quality,
)
from ..lod import active_tier
from ..procedural.sdf import op_smooth_union, sd_sphere
from ..scene import Scene

_MAXD = 30.0


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    plane = p[1] + 1.0
    bob = 0.15 * wp.sin(time * 1.3)
    s0 = sd_sphere(p - wp.vec3(-1.7, bob, 0.0), 0.7)
    s1 = sd_sphere(p - wp.vec3(0.0, -bob, 0.0), 0.7)
    s2 = sd_sphere(p - wp.vec3(1.7, bob, 0.0), 0.7)
    spheres = wp.min(s0, wp.min(s1, s2))
    return op_smooth_union(plane, spheres, 0.25)


@wp.func
def _mat_id(p: wp.vec3, time: float) -> int:
    bob = 0.15 * wp.sin(time * 1.3)
    d0 = sd_sphere(p - wp.vec3(-1.7, bob, 0.0), 0.7)
    d1 = sd_sphere(p - wp.vec3(0.0, -bob, 0.0), 0.7)
    d2 = sd_sphere(p - wp.vec3(1.7, bob, 0.0), 0.7)
    plane = p[1] + 1.0
    best = plane
    which = int(0)
    if d0 < best:
        best = d0
        which = 1
    if d1 < best:
        best = d1
        which = 2
    if d2 < best:
        best = d2
        which = 3
    return which


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, time: float, steps: int) -> float:
    res = float(1.0)
    t = float(0.04)
    for _ in range(steps):
        h = _map(ro + rd * t, time)
        if h < 0.001:
            return 0.0
        res = wp.min(res, 10.0 * h / t)
        t += wp.clamp(h, 0.02, 0.3)
        if t > 12.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float, taps: int) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(taps):
        hr = 0.02 + 0.12 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.5 * occ, 0.0, 1.0)


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.55, 0.62, 0.78) * (1.0 - up) + wp.vec3(0.14, 0.30, 0.62) * up
    s = wp.pow(wp.max(wp.dot(rd, sun), 0.0), 64.0)
    return base + wp.vec3(1.0, 0.9, 0.7) * (s * 4.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, light: Light,
                  qual: Quality, frame: Frame):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(frame.width)) - 1.0
    v = (2.0 * (float(frame.height - 1 - i) + 0.5) / float(frame.height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    time = frame.time

    # adaptive sphere tracing
    t = float(0.0)
    hit = int(0)
    for _ in range(qual.raymarch_steps):
        p = ro + rd * t
        d = _map(p, time)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.9
        if t > _MAXD:
            break

    if hit == 0:
        col = _sky(rd, light.dir)
    else:
        p = ro + rd * t
        n = _normal(p, time)
        mid = _mat_id(p, time)
        # per-material albedo / roughness / metallic
        albedo = wp.vec3(0.6, 0.6, 0.6)
        rough = float(0.9)
        metallic = float(0.0)
        if mid == 0:  # ground: checker
            chk = wp.floor(p[0]) + wp.floor(p[2])
            g = 0.2 + 0.15 * wp.abs(chk - 2.0 * wp.floor(chk * 0.5))
            albedo = wp.vec3(g, g, g)
            rough = 0.85
        elif mid == 1:
            albedo = wp.vec3(0.85, 0.20, 0.18)
            rough = 0.15
        elif mid == 2:
            albedo = wp.vec3(0.95, 0.78, 0.35)
            rough = 0.30
            metallic = 1.0
        else:
            albedo = wp.vec3(0.20, 0.45, 0.85)
            rough = 0.6

        v_dir = -rd
        sh = _soft_shadow(p + n * 0.01, light.dir, time, qual.shadow_steps)
        ao = _ao(p, n, time, qual.ao_steps)
        mat = make_mat(albedo, rough, metallic)
        direct = shade_material(mat, n, v_dir, light.dir, light.color, light.intensity * sh)
        # ambient from sky
        amb = wp.cw_mul(_sky(n, light.dir), albedo) * (0.25 * ao)
        col = direct + amb

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = active_tier()
    az = 0.6 + 0.25 * math.sin(time * 0.2) + float(mouse[0]) * 0.01
    el = 0.35 + float(mouse[1]) * 0.005
    dist = 5.0
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el) + 0.5,
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, -0.1, 0.0), fov_deg=42.0, aspect=width / height)
    light = make_light((0.5, 0.75, 0.4), (1.0, 0.96, 0.9), 3.2)
    qual = make_quality(tier)
    frame = make_frame(time, width, height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, light, qual, frame], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=2)
    out = post.tonemap(hdr, mode="aces", exposure=1.15)
    return post.vignette(out, 0.25)


SCENE = Scene(
    name="pbr_demo",
    description="Raymarched GGX-PBR scene (adaptive LOD, soft shadows, AO, bloom). --quality low..ultra.",
    renderer=_render,
)
