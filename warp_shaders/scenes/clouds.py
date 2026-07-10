"""Volumetric cloudscape — raymarched clouds over an analytic day sky.

Proves the volumetric engine: Schneider-style density, Henyey-Greenstein phase,
Beer-Lambert extinction, sun light-march self-shadowing, powder edges. Cloud
march steps scale with `--quality`. iMouse: x = look azimuth, y = sun height.
"""

import math

import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..engine.volumetric import march_clouds
from ..lod import active_tier
from ..scene import Scene

_BASE = 60.0
_TOP = 165.0


def _cloud_samples(tier_name):
    return {"low": (24, 4), "medium": (40, 6), "high": (64, 8),
            "ultra": (96, 12)}.get(tier_name, (40, 6))


@wp.func
def _skybg(rd: wp.vec3, sun: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.6 + 0.4, 0.0, 1.0)
    base = wp.vec3(0.72, 0.80, 0.92) * (1.0 - up) + wp.vec3(0.18, 0.42, 0.86) * up
    s = wp.max(wp.dot(rd, sun), 0.0)
    glow = wp.pow(s, 8.0) * 0.25 + wp.pow(s, 512.0) * 10.0
    return base + wp.vec3(1.0, 0.9, 0.7) * glow


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  time: float, coverage: float, steps: int, light_steps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = camera_ray_dir(cam, u, v)

    sky = _skybg(rd, sun)
    col = sky
    if rd[1] > 0.001:
        sun_col = wp.vec3(1.0, 0.95, 0.85)
        amb = wp.vec3(0.40, 0.52, 0.72) * 0.32
        c = march_clouds(cam.eye, rd, sun, time, coverage, _BASE, _TOP,
                         steps, light_steps, sun_col, amb)
        col = sky * c[3] + wp.vec3(c[0], c[1], c[2])
    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = active_tier()
    steps, lsteps = _cloud_samples(tier.name)

    az = 0.4 + float(mouse[0]) * 0.01 + time * 0.02
    pitch = 0.14
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    eye = (0.0, 10.0, 0.0)
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=64.0, aspect=width / height)

    el = 0.55 + float(mouse[1]) * 0.004
    sun = wp.vec3(math.sin(az * 0.6) * math.cos(el), math.sin(el), math.cos(az * 0.6) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time), 0.58, int(steps), int(lsteps),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(3, int(min(width, height) * 0.015))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.35, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="clouds",
    description="Volumetric cloudscape (HG phase, Beer-Lambert, light-march). --quality low..ultra.",
    renderer=_render,
)
