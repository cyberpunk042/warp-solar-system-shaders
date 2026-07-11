"""Procedural terrain — a hyper-real raymarched landscape.

A heightfield raymarcher over ridged+fbm mountains, shaded with slope/height
materials (grass / rock / snow), IQ soft shadows, sky ambient, an analytic sky +
sun, and aerial-perspective distance fog. Exercises the whole engine on a new
subject (not a globe). March/shadow steps scale with `--quality`. iMouse pans.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm_perlin3, ridged3
from ..scene import Scene

_FAR = 320.0


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.018, 0.0, z * 0.018)
    hills = fbm_perlin3(p, 6) * 7.0
    mountains = ridged3(p * 0.55, 6) * 26.0
    # ridged crests dominate the highs; hills add mid detail
    return hills + mountains - 6.0


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.08
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.6 + 0.4, 0.0, 1.0)
    base = wp.vec3(0.62, 0.72, 0.86) * (1.0 - up) + wp.vec3(0.16, 0.36, 0.78) * up
    s = wp.max(wp.dot(rd, sun), 0.0)
    return base + wp.vec3(1.0, 0.85, 0.6) * (wp.pow(s, 8.0) * 0.3 + wp.pow(s, 900.0) * 12.0)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.5)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2])
        if h < 0.001:
            return 0.0
        res = wp.min(res, 12.0 * h / t)
        t += wp.clamp(h, 0.4, 8.0)
        if t > 120.0:
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

    # heightfield raymarch: step until the ray drops below the surface, then
    # bisect for the exact crossing (removes step banding, allows big steps)
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
        t += wp.max(d * 0.4, 0.01 * t)
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
    slope = n[1]                       # 1 = flat
    hnorm = wp.clamp((p[1] + 6.0) / 34.0, 0.0, 1.0)

    grass = wp.vec3(0.09, 0.26, 0.08)
    rock = wp.vec3(0.28, 0.25, 0.22)
    snow = wp.vec3(0.92, 0.94, 0.98)
    # rock on steep slopes, grass on gentle low ground, snow up high
    rocky = wp.smoothstep(0.75, 0.55, slope)
    albedo = grass * (1.0 - rocky) + rock * rocky
    # snow: mostly altitude-driven, only shed on the very steepest faces
    snowf = wp.smoothstep(0.62, 0.82, hnorm) * wp.smoothstep(0.3, 0.55, slope)
    albedo = albedo * (1.0 - snowf) + snow * snowf

    v_dir = -rd
    sh = _shadow(p + n * 0.05, sun, shadow_steps)
    direct = shade_pbr(n, v_dir, sun, albedo, 0.85, 0.0, wp.vec3(1.0, 0.93, 0.82)) * (2.6 * sh)
    amb = wp.cw_mul(_sky(n, sun), albedo) * (0.35 * (0.5 + 0.5 * n[1]))
    col = direct + amb

    # aerial perspective: fade to sky with distance
    fog = 1.0 - wp.exp(-t * 0.006)
    col = col * (1.0 - fog) + _sky(rd, sun) * fog
    img[i, j] = col


def _counts(name):
    return {"low": (120, 16), "medium": (200, 24), "high": (300, 32),
            "ultra": (440, 48)}.get(name, (200, 24))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.5 + float(mouse[0]) * 0.008 + time * 0.02
    eye = (math.sin(az) * 6.0, 9.0, math.cos(az) * 6.0)
    target = (eye[0] + math.sin(az) * 10.0, 6.0 + float(mouse[1]) * 0.02,
              eye[2] + math.cos(az) * 10.0)
    cam = make_camera(eye, target, fov_deg=62.0, aspect=width / height)

    el = 0.34
    sun = wp.vec3(math.sin(az + 1.3) * math.cos(el), math.sin(el), math.cos(az + 1.3) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.014))
    hdr = post.bloom(hdr, threshold=1.4, strength=0.35, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="terrain",
    description="Raymarched procedural landscape (mountains, materials, soft shadows, aerial perspective). --quality low..ultra.",
    renderer=_render,
)
