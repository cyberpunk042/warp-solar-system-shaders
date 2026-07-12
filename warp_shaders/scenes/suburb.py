"""A suburb — a neighbourhood of pitched-roof houses, ray-marched.

Sphere-traces `buildings.suburb_de` (a grid of houses from one function via
domain repetition + hashed variation) on a grassy ground, with plaster walls,
tiled roofs, a warm low sun, soft shadows, sky and fog. The human-scale
counterpart to `city`, and — like it — a target the blast can later be tested on.
See ``docs/research/17-buildings.md``.
"""

import math

import warp as wp

from ..buildings.sdf import suburb_de
from ..engine import post
from ..engine.pbr import shade_pbr
from ..engine.shading import apply_fog, sky_gradient, sun_disk
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..scene import Scene

_LOT = wp.constant(9.0)
_SEED = wp.constant(5.0)


@wp.func
def _map(p: wp.vec3) -> float:
    return wp.min(suburb_de(p, _LOT, _SEED)[0], p[1])


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.015
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    base = sky_gradient(rd, wp.vec3(0.95, 0.82, 0.66), wp.vec3(0.30, 0.52, 0.86))
    return base + sun_disk(rd, sun, wp.vec3(1.0, 0.9, 0.7), 0.9995, 0.5)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.15)
    for _ in range(steps):
        h = _map(p + sun * t)
        if h < 0.002:
            return 0.0
        res = wp.min(res, 16.0 * h / t)
        t += wp.clamp(h, 0.04, 2.0)
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

    t = float(0.0)
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = _map(p)
        if d < 0.0015 * t + 0.003:
            hit = 1
            break
        t += d * 0.6
        if t > 240.0:
            break

    col = _sky(rd, sun)
    if hit == 1:
        p = ro + rd * t
        n = _normal(p)
        v4 = suburb_de(p, _LOT, _SEED)
        is_ground = wp.step(v4[0] - 0.05)                     # on the lawn, not a house
        # roof is the geometry above twice the body half-height
        roof_m = wp.step(p[1] - 2.0 * v4[1] * 0.92) * (1.0 - is_ground)
        lr = v4[3]
        plaster = wp.vec3(0.80, 0.70, 0.56) * (0.7 + 0.5 * lr)     # warm walls, varied
        tile = wp.vec3(0.62, 0.22, 0.14) * (0.85 + 0.35 * v4[2])   # terracotta roof
        grass = wp.vec3(0.13, 0.26, 0.10)
        wall = plaster * (1.0 - roof_m) + tile * roof_m
        albedo = wall * (1.0 - is_ground) + grass * is_ground
        sh = _shadow(p + n * 0.02, sun, shadow_steps)
        direct = shade_pbr(n, -rd, sun, albedo, 0.85, 0.0, wp.vec3(1.0, 0.86, 0.64)) * (3.1 * sh)
        # warm neutral hemispheric ambient (not the blue sky, which muddied it)
        amb = wp.cw_mul(albedo, wp.vec3(0.40, 0.37, 0.33)) * (0.45 + 0.4 * n[1])
        col = apply_fog(direct + amb, t, _sky(rd, sun), 0.0022)

    img[i, j] = col


def _counts(name):
    return {"low": (110, 18), "medium": (170, 28), "high": (240, 40),
            "ultra": (330, 60)}.get(name, (170, 28))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)
    az = 0.6 + time * 0.05 + float(mouse[0]) * 0.01
    dist = 27.0
    eye = (math.sin(az) * dist, 15.0 + float(mouse[1]) * 0.05, math.cos(az) * dist)
    cam = make_camera(eye, (0.0, 1.2, 0.0), fov_deg=52.0, aspect=width / height)
    el = 0.34
    sun = wp.normalize(wp.vec3(math.cos(az + 1.3) * math.cos(el), math.sin(el),
                               math.sin(az + 1.3) * math.cos(el)))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.3, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="suburb",
    description="A neighbourhood of SDF houses (domain-repeated, hashed variation) "
                "— plaster walls, terracotta pitched roofs, warm low sun.",
    renderer=_render,
)
