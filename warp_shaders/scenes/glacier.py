"""Glacier — a crystalline ice field under a cold low sun.

A jagged heightfield of ridged ice seracs with fine crystal facets, shaded as
two materials: glossy blue ice on the steep faces (low roughness + a faked
subsurface blue glow) and rougher snow on the gentle ground, with a hash-driven
snow sparkle facing the light. Cold low sun for long blue shadows, cold aerial
haze, pale-blue sky. Heightfield raymarch with crossing-detection + bisection;
`--quality` scales march/shadow steps. iMouse pans.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sky_gradient, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.hash import hash31
from ..procedural.noise import fbm_perlin3, ridged3
from ..scene import Scene

_FAR = 320.0


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.02, 0.0, z * 0.02)
    base = fbm_perlin3(p, 6) * 6.0
    crags = ridged3(p * 0.7, 6) * 17.0          # sharp ice seracs
    facets = ridged3(p * 3.1, 4) * 1.1          # crystalline facet detail
    return base + crags + facets - 5.0


@wp.func
def _normal(x: float, z: float) -> wp.vec3:
    e = 0.06
    nx = _height(x - e, z) - _height(x + e, z)
    nz = _height(x, z - e) - _height(x, z + e)
    return wp.normalize(wp.vec3(nx, 2.0 * e, nz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    base = sky_gradient(rd, wp.vec3(0.78, 0.84, 0.92), wp.vec3(0.33, 0.5, 0.76))
    return base + sun_disk(rd, sun, wp.vec3(0.9, 0.95, 1.0), 0.9991, 0.4)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.4)
    for _ in range(steps):
        q = p + sun * t
        h = q[1] - _height(q[0], q[2])
        if h < 0.001:
            return 0.0
        res = wp.min(res, 12.0 * h / t)
        t += wp.clamp(h, 0.35, 6.0)
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
        t += wp.max(d * 0.45, 0.01 * t)
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
    slope = n[1]                        # 1 = flat

    # steep faces = glossy blue ice; gentle ground = rough snow
    icy = wp.smoothstep(0.82, 0.55, slope)
    ice = wp.vec3(0.32, 0.52, 0.74)
    snow = wp.vec3(0.86, 0.91, 0.99)
    albedo = snow * (1.0 - icy) + ice * icy
    rough = 0.55 * (1.0 - icy) + 0.15 * icy

    sh = _shadow(p + n * 0.04, sun, shadow_steps)
    sun_col = wp.vec3(1.0, 0.98, 0.92)
    direct = shade_pbr(n, v_dir, sun, albedo, rough, 0.0, sun_col) * (2.7 * sh)
    # cold sky ambient + faked subsurface blue glow inside the ice faces
    amb = wp.cw_mul(_sky(n, sun), albedo) * (0.26 * (0.5 + 0.5 * n[1]))
    sss = wp.vec3(0.1, 0.24, 0.48) * (icy * (0.4 + 0.5 * slope))
    # snow sparkle: sparse hash-driven glints on the gentle ground facing the sun
    sparkle = float(0.0)
    if icy < 0.4:
        g = hash31(wp.vec3(p[0] * 9.0, p[1] * 9.0, p[2] * 9.0))
        if g > 0.985:
            sparkle = wp.max(wp.dot(n, sun), 0.0) * sh * 3.0
    col = direct + amb + sss + wp.vec3(sparkle, sparkle, sparkle)

    col = apply_fog(col, t, wp.vec3(0.66, 0.75, 0.86), 0.0028)
    img[i, j] = col


def _counts(name):
    return {"low": (120, 16), "medium": (200, 24), "high": (300, 32),
            "ultra": (440, 48)}.get(name, (200, 24))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)

    az = 0.6 + float(mouse[0]) * 0.008 + time * 0.02
    eye = (math.sin(az) * 6.0, 9.0, math.cos(az) * 6.0)
    target = (eye[0] + math.sin(az) * 10.0, 6.0 + float(mouse[1]) * 0.02,
              eye[2] + math.cos(az) * 10.0)
    cam = make_camera(eye, target, fov_deg=62.0, aspect=width / height)

    el = 0.16                          # low cold sun
    sun = wp.vec3(math.sin(az + 1.9) * math.cos(el), math.sin(el),
                  math.cos(az + 1.9) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.4, strength=0.3, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=0.82)


SCENE = Scene(
    name="glacier",
    description="Crystalline ice field: blue ice + snow, subsurface glow, cold low sun. --quality low..ultra.",
    renderer=_render,
)
