"""Ocean — a PBR water surface with analytic waves.

A sum-of-sines wave field (analytic normals, no finite differences), raymarched
with crossing + bisection. Shaded as water: Fresnel-blended sky reflection + deep
water colour, a GGX sun-glitter specular, and foam on the crests. Analytic sky +
sun behind. `--quality` scales march/reflection detail. iMouse pans / lowers sun.
"""

import math

import warp as wp

from ..engine import post
from ..engine.pbr import distribution_ggx, fresnel_schlick
from ..engine.shading import apply_fog
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..scene import Scene

_FAR = 600.0


@wp.func
def _wave_h(x: float, z: float, t: float) -> float:
    h = 0.28 * wp.sin(x * 0.30 + z * 0.10 + t * 0.9)
    h += 0.18 * wp.sin(x * 0.12 - z * 0.38 + t * 1.1)
    h += 0.12 * wp.sin(-x * 0.45 + z * 0.22 + t * 1.6)
    h += 0.07 * wp.sin(x * 0.75 + z * 0.60 + t * 2.2)
    h += 0.04 * wp.sin(x * 1.20 - z * 0.90 + t * 3.0)
    return h


@wp.func
def _wave_n(x: float, z: float, t: float) -> wp.vec3:
    dx = 0.28 * 0.30 * wp.cos(x * 0.30 + z * 0.10 + t * 0.9)
    dz = 0.28 * 0.10 * wp.cos(x * 0.30 + z * 0.10 + t * 0.9)
    dx += 0.18 * 0.12 * wp.cos(x * 0.12 - z * 0.38 + t * 1.1)
    dz += 0.18 * (-0.38) * wp.cos(x * 0.12 - z * 0.38 + t * 1.1)
    dx += 0.12 * (-0.45) * wp.cos(-x * 0.45 + z * 0.22 + t * 1.6)
    dz += 0.12 * 0.22 * wp.cos(-x * 0.45 + z * 0.22 + t * 1.6)
    dx += 0.07 * 0.75 * wp.cos(x * 0.75 + z * 0.60 + t * 2.2)
    dz += 0.07 * 0.60 * wp.cos(x * 0.75 + z * 0.60 + t * 2.2)
    dx += 0.04 * 1.20 * wp.cos(x * 1.20 - z * 0.90 + t * 3.0)
    dz += 0.04 * (-0.90) * wp.cos(x * 1.20 - z * 0.90 + t * 3.0)
    return wp.normalize(wp.vec3(-dx, 1.0, -dz))


@wp.func
def _sky(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.7 + 0.3, 0.0, 1.0)
    base = wp.vec3(0.68, 0.78, 0.9) * (1.0 - up) + wp.vec3(0.13, 0.33, 0.72) * up
    s = wp.max(wp.dot(rd, sun), 0.0)
    return base + wp.vec3(1.0, 0.85, 0.62) * (wp.pow(s, 6.0) * 0.4 + wp.pow(s, 1200.0) * 16.0)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  time: float, march_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    if rd[1] > -0.001:
        img[i, j] = _sky(rd, sun)
        return

    # heightfield raymarch to the wave surface (crossing + bisection)
    t = float(0.5)
    prev_t = t
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        if p[1] - _wave_h(p[0], p[2], time) < 0.0:
            hit = 1
            break
        prev_t = t
        d = p[1] - _wave_h(p[0], p[2], time)
        t += wp.max(d * 0.5, 0.02 * t)
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
        if pm[1] - _wave_h(pm[0], pm[2], time) < 0.0:
            b = m
        else:
            a = m
    t = 0.5 * (a + b)
    p = ro + rd * t
    n = _wave_n(p[0], p[2], time)

    v_dir = -rd
    # sky reflection
    refl = rd - n * (2.0 * wp.dot(rd, n))
    refl = wp.vec3(refl[0], wp.abs(refl[1]) + 0.01, refl[2])
    sky_refl = _sky(wp.normalize(refl), sun)
    # Fresnel (water F0 ~ 0.02)
    fres = fresnel_schlick(wp.max(wp.dot(n, v_dir), 0.0), wp.vec3(0.02, 0.02, 0.02))
    deep = wp.vec3(0.0, 0.09, 0.13)
    shallow = wp.vec3(0.0, 0.22, 0.28)
    water = deep * (1.0 - wp.max(n[1], 0.0)) + shallow * wp.max(n[1], 0.0)
    col = water * (1.0 - fres[2]) + sky_refl * fres[2]

    # GGX sun-glitter
    h = wp.normalize(v_dir + sun)
    ndh = wp.max(wp.dot(n, h), 0.0)
    spec = distribution_ggx(ndh, 0.06) * wp.max(wp.dot(n, sun), 0.0)
    col = col + wp.vec3(1.0, 0.95, 0.8) * (spec * 1.5)

    # foam on steep crests
    foam = wp.smoothstep(0.55, 0.72, _wave_h(p[0], p[2], time) + 0.35) * (1.0 - n[1]) * 3.0
    col = col + wp.vec3(0.9, 0.94, 0.98) * wp.clamp(foam, 0.0, 0.5)

    # distance haze to the horizon
    col = apply_fog(col, t, _sky(rd, sun), 0.004)
    img[i, j] = col


def _march(name):
    return {"low": 60, "medium": 100, "high": 160, "ultra": 240}.get(name, 100)


def _render(width, height, time, mouse, device):
    tier = active_tier()
    ms = _march(tier.name)

    az = 0.2 + float(mouse[0]) * 0.008
    pitch = -0.12
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    eye = (0.0, 2.2, 0.0)
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=66.0, aspect=width / height)

    el = 0.09 + float(mouse[1]) * 0.003
    sun = wp.vec3(math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time), int(ms), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.018))
    hdr = post.bloom(hdr, threshold=1.5, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="ocean",
    description="PBR ocean: analytic waves, Fresnel sky reflection, GGX sun-glitter, foam. --quality low..ultra.",
    renderer=_render,
)
