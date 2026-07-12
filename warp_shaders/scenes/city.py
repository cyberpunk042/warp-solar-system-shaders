"""A city — a skyline of SDF buildings, ray-marched.

Sphere-traces `buildings.city_de` (a whole city from one function via per-lot
domain repetition + hashed variation) over a street grid, shading concrete vs
glass from `window_mask`, with the engine's sun, soft shadows, sky and aerial
fog (which also hides the far-distance repetition flicker). A low sun rakes long
shadows down the avenues. See ``docs/research/17-buildings.md``. `--quality`
scales the march / shadow steps.
"""

import math

import warp as wp

from ..buildings.sdf import city_de, window_mask
from ..engine import post
from ..engine.shading import apply_fog
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.hash import hash21
from ..scene import Scene

_LOT = wp.constant(16.0)
_SEED = wp.constant(3.0)


@wp.func
def _map(p: wp.vec3) -> float:
    # buildings (base on the ground) unioned with the y=0 street plane
    b = city_de(p, _LOT, _SEED)[0]
    return wp.min(b, p[1])


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.02
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _sky(rd: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.03, 0.04, 0.08) * (1.0 - up) + wp.vec3(0.01, 0.02, 0.05) * up
    # warm horizon glow — city light pollution on the low sky
    horizon = wp.pow(1.0 - wp.clamp(rd[1] + 0.06, 0.0, 1.0), 6.0)
    base = base + wp.vec3(0.30, 0.18, 0.09) * (horizon * 0.6)
    # stars
    s = hash21(wp.vec2(wp.floor(rd[0] * 190.0), wp.floor(rd[2] * 190.0)))
    star = wp.step(s - 0.994) * wp.clamp(rd[1], 0.0, 1.0)
    return base + wp.vec3(star, star, star)


@wp.func
def _shadow(p: wp.vec3, sun: wp.vec3, steps: int) -> float:
    res = float(1.0)
    t = float(0.2)
    for _ in range(steps):
        h = _map(p + sun * t)
        if h < 0.002:
            return 0.0
        res = wp.min(res, 14.0 * h / t)
        t += wp.clamp(h, 0.05, 3.0)
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

    t = float(0.0)
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = _map(p)
        if d < 0.002 * t + 0.004:
            hit = 1
            break
        t += d * 0.6                                  # small step: domain-rep field
        if t > 600.0:
            break

    col = _sky(rd)
    if hit == 1:
        p = ro + rd * t
        n = _normal(p)
        v4 = city_de(p, _LOT, _SEED)                          # lot rand for variety
        is_ground = wp.step(v4[0] - 0.06)                     # ground plane vs building
        lr = v4[3]
        win = window_mask(p, 1.1, 1.6) * (1.0 - is_ground)
        # per-window lit hash (floor x column x lot) — ~half the panes glow
        fi = wp.floor(p[1] / 1.6)
        ci = wp.floor((p[0] + p[2]) / 1.1)
        lh = hash21(wp.vec2(ci + lr * 61.0, fi - lr * 37.0))
        lit = win * wp.step(lh - 0.5)
        # night materials
        concrete = wp.vec3(0.05, 0.05, 0.07)
        glassd = wp.vec3(0.03, 0.04, 0.06)
        street = wp.vec3(0.045, 0.045, 0.055)
        mat = concrete * (1.0 - win) + glassd * win
        mat = mat * (1.0 - is_ground) + street * is_ground
        # dim moonlight + cool sky ambient
        ndl = wp.max(wp.dot(n, sun), 0.0)
        amb = wp.cw_mul(mat, wp.vec3(0.32, 0.42, 0.65)) * (0.22 + 0.55 * ndl)
        # emissive lit windows — warm, with slight per-window colour variation
        wc = wp.vec3(1.0, 0.78, 0.44) + wp.vec3(0.0, 0.10, 0.30) * (lh - 0.5)
        emit = wc * (lit * 2.7)
        col = amb + emit
        col = apply_fog(col, t, _sky(rd), 0.004)

    img[i, j] = col


def _counts(name):
    return {"low": (110, 20), "medium": (170, 30), "high": (240, 44),
            "ultra": (340, 64)}.get(name, (170, 30))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms, ss = _counts(tier.name)
    az = 0.6 + time * 0.03 + float(mouse[0]) * 0.01
    dist = 120.0
    eye = (math.sin(az) * dist, 34.0 + float(mouse[1]) * 0.1, math.cos(az) * dist)
    cam = make_camera(eye, (0.0, 15.0, 0.0), fov_deg=54.0, aspect=width / height)
    moon = wp.normalize(wp.vec3(math.cos(az + 1.4) * 0.6, 0.6, math.sin(az + 1.4) * 0.6))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, moon, int(ms), int(ss), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.6, strength=0.55, radius=r, passes=3)     # bloom the windows
    return post.tonemap(hdr, mode="aces", exposure=1.02)


SCENE = Scene(
    name="city",
    description="A skyline of SDF buildings (domain-repeated towers/blocks with "
                "hashed variation) — concrete + glass, low sun, long shadows.",
    renderer=_render,
)
