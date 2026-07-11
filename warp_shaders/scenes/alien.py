"""Alien world — a heightfield landscape under twin suns and a colored sky.

Reuses the terrain heightfield-raymarch pattern (crossing + bisection) with jagged
ridged peaks, an exotic purple/teal/ice palette, two coloured suns (warm + cool)
each casting soft shadows, a violet-amber atmosphere with a large moon, and
coloured aerial-perspective fog. `--quality` scales march/shadow steps.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm_perlin3, ridged3
from ..scene import Scene

_FAR = 340.0
_MOON = wp.constant(wp.vec3(-0.45, 0.5, 0.74))


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.02, 0.0, z * 0.02)
    ridges = ridged3(p * 0.6, 6) * 34.0        # jagged spires
    hills = fbm_perlin3(p, 5) * 6.0
    return ridges + hills - 10.0


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.08
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _sky(rd: wp.vec3, s1: wp.vec3, s2: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.6 + 0.4, 0.0, 1.0)
    base = wp.vec3(0.55, 0.30, 0.28) * (1.0 - up) + wp.vec3(0.28, 0.16, 0.42) * up
    g1 = wp.max(wp.dot(rd, s1), 0.0)
    g2 = wp.max(wp.dot(rd, s2), 0.0)
    base = base + wp.vec3(1.0, 0.55, 0.25) * (wp.pow(g1, 6.0) * 0.4 + wp.pow(g1, 1500.0) * 14.0)
    base = base + wp.vec3(0.4, 0.6, 1.0) * (wp.pow(g2, 6.0) * 0.3 + wp.pow(g2, 1500.0) * 10.0)
    # large moon
    md = wp.max(wp.dot(rd, wp.normalize(_MOON)), 0.0)
    moon = wp.smoothstep(0.995, 0.997, md)
    mshade = 0.4 + 0.6 * wp.max(wp.dot(wp.normalize(_MOON), s1), 0.0)
    return base + wp.vec3(0.8, 0.8, 0.9) * (moon * mshade)


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
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, s1: wp.vec3, s2: wp.vec3,
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
        t += wp.max(d * 0.4, 0.02 * t)
        if t > _FAR:
            break

    if hit == 0:
        img[i, j] = _sky(rd, s1, s2)
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
    slope = n[1]
    hnorm = wp.clamp((p[1] + 10.0) / 40.0, 0.0, 1.0)

    rock = wp.vec3(0.34, 0.26, 0.42)
    low = wp.vec3(0.12, 0.34, 0.30)
    ice = wp.vec3(0.80, 0.86, 0.98)
    rocky = wp.smoothstep(0.8, 0.55, slope)
    albedo = low * (1.0 - rocky) + rock * rocky
    icef = wp.smoothstep(0.68, 0.85, hnorm) * wp.smoothstep(0.35, 0.6, slope)
    albedo = albedo * (1.0 - icef) + ice * icef

    v_dir = -rd
    sh1 = _shadow(p + n * 0.05, s1, shadow_steps)
    sh2 = _shadow(p + n * 0.05, s2, shadow_steps)
    c1 = shade_pbr(n, v_dir, s1, albedo, 0.8, 0.0, wp.vec3(1.0, 0.55, 0.25)) * (3.4 * sh1)
    c2 = shade_pbr(n, v_dir, s2, albedo, 0.8, 0.0, wp.vec3(0.35, 0.55, 1.0)) * (2.2 * sh2)
    amb = wp.cw_mul(_sky(n, s1, s2), albedo) * (0.18 * (0.5 + 0.5 * n[1]))
    col = c1 + c2 + amb

    fog = 1.0 - wp.exp(-t * 0.0042)
    col = col * (1.0 - fog) + _sky(rd, s1, s2) * fog
    img[i, j] = col


def _counts(name):
    return {"low": (120, 16), "medium": (200, 24), "high": (300, 32),
            "ultra": (440, 48)}.get(name, (200, 24))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.5 + float(mouse[0]) * 0.008 + time * 0.02
    eye = (math.sin(az) * 6.0, 11.0, math.cos(az) * 6.0)
    target = (eye[0] + math.sin(az) * 10.0, 7.0 + float(mouse[1]) * 0.02,
              eye[2] + math.cos(az) * 10.0)
    cam = make_camera(eye, target, fov_deg=64.0, aspect=width / height)

    s1 = wp.vec3(math.sin(az + 1.0) * 0.9, 0.28, math.cos(az + 1.0) * 0.9)
    s2 = wp.vec3(math.sin(az - 1.6) * 0.8, 0.16, math.cos(az - 1.6) * 0.8)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, s1, s2, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.014))
    hdr = post.bloom(hdr, threshold=1.4, strength=0.4, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="alien",
    description="Alien landscape: jagged terrain, twin coloured suns, violet sky + moon. --quality low..ultra.",
    renderer=_render,
)
