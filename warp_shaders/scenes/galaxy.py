"""Galaxy — a volumetric emissive spiral disk.

Raymarches a thin galactic disk: log-spiral arms (blue-white), a warm core bulge,
and pink star-forming knots, with a vertical gaussian profile and radial falloff.
Emissive accumulation with mild dust extinction, over a starfield, viewed at an
inclination. `--quality` scales the march steps. iMouse orbits.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm3
from ..scene import Scene

_RD = 2.6
_TH = 0.16


@wp.func
def _slab(ro: wp.vec3, rd: wp.vec3) -> wp.vec2:
    if wp.abs(rd[1]) < 1e-5:
        return wp.vec2(1.0, -1.0)
    t0 = (-_TH - ro[1]) / rd[1]
    t1 = (_TH - ro[1]) / rd[1]
    return wp.vec2(wp.min(t0, t1), wp.max(t0, t1))


@wp.func
def _emit(p: wp.vec3, time: float) -> wp.vec3:
    r = wp.length(wp.vec2(p[0], p[2]))
    if r > _RD:
        return wp.vec3(0.0, 0.0, 0.0)
    th = wp.atan2(p[2], p[0]) + time * 0.05
    turb = fbm3(p * 2.5, 3)
    arms = wp.sin(2.0 * th + 7.0 * wp.log(r + 0.25) + turb * 2.2)
    arm = wp.pow(wp.max(arms, 0.0), 2.2)
    vert = wp.exp(-(p[1] * p[1]) / 0.010)
    rad = wp.exp(-r * 1.15)
    core = wp.exp(-r * r * 9.0)

    disk = arm * rad * vert
    arm_col = wp.vec3(0.6, 0.75, 1.0) * (disk * 1.8)
    core_col = wp.vec3(1.0, 0.86, 0.55) * (core * 3.2)
    hii = wp.smoothstep(0.7, 0.92, turb) * disk
    hii_col = wp.vec3(1.0, 0.35, 0.6) * (hii * 3.5)
    return (arm_col + core_col + hii_col) * (0.6 + 0.6 * turb)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                  steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = stars(rd)
    sb = _slab(ro, rd)
    t0 = wp.max(sb[0], 0.0)
    if sb[1] > t0:
        seg = (sb[1] - t0) / float(steps)
        acc = wp.vec3(0.0, 0.0, 0.0)
        trans = float(1.0)
        t = t0 + 0.5 * seg
        for _ in range(steps):
            p = ro + rd * t
            e = _emit(p, time)
            dens = (e[0] + e[1] + e[2]) * 0.2
            acc += e * (seg * 1.4 * trans)
            trans *= wp.exp(-dens * seg * 0.5)
            if trans < 0.05:
                break
            t += seg
        col = col * trans + acc
    img[i, j] = col


def _steps(name):
    return {"low": 40, "medium": 64, "high": 100, "ultra": 160}.get(name, 64)


def _render(width, height, time, mouse, device):
    tier = active_tier()
    steps = _steps(tier.name)

    az = 0.4 + float(mouse[0]) * 0.01 + time * 0.03
    el = 0.5 + float(mouse[1]) * 0.004      # inclination
    dist = 6.5
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=40.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(steps), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.7, strength=0.7, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.2)


SCENE = Scene(
    name="galaxy",
    description="Volumetric spiral galaxy: log-spiral arms, core bulge, star-forming knots. --quality low..ultra.",
    renderer=_render,
)
