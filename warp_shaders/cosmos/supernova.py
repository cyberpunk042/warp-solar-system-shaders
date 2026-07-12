"""Supernova shockwave — a core-collapse explosion and its expanding shell.

Drives the existing supernova envelope (`stellar_evolution._march_env` with
`ENV_SUPERNOVA` — incandescent orange-red ejecta behind a blue-white leading
shock, with Rayleigh-Taylor filaments) with a **growing radius** and an initial
blinding **flash**, plus the hot **remnant core** left at the centre. The shell
dims as it expands (blackbody cooling). See
``docs/research/20-more-cosmos-worlds-crossstrand.md``.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from .stellar_evolution import ENV_SUPERNOVA, _march_env


@wp.kernel
def supernova_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, radius: float,
                     core_k: float, core_glow: float, intensity: float, seed: float,
                     time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    col = stars(rd)
    # hot remnant core at the origin — impact parameter of the ray to the centre
    bimp = wp.length(wp.cross(ro, rd))                   # rd is unit
    core = wp.exp(-(bimp * bimp) / 0.25) * core_glow
    col = col + kelvin_to_rgb(core_k) * (core * 3.0)
    # expanding shock-heated shell
    col = col + _march_env(ro, rd, ENV_SUPERNOVA, radius, intensity, seed, time)
    img[i, j] = col


def render_supernova(width, height, time, mouse, device, period=12.0):
    """Render one frame of a supernova — flash, then an expanding cooling shell."""
    prog = time / period
    radius = 0.4 + 9.5 * min(prog, 1.0)                  # the shell expands
    core_k = 1500.0 + 26000.0 * math.exp(-1.8 * prog)   # remnant cools from blue-white
    core_glow = max(0.15, 1.4 * math.exp(-1.2 * prog))
    intensity = 3.2 / (1.0 + 0.32 * radius)             # shell dims as it thins out

    az = 0.5 + float(mouse[0]) * 0.01
    elev = 0.18 + float(mouse[1]) * 0.01
    dist = 16.0
    eye = (math.sin(az) * dist * math.cos(elev), math.sin(elev) * dist,
           math.cos(az) * dist * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=52.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(supernova_kernel, dim=(height, width),
              inputs=[img, cam, float(radius), float(core_k), float(core_glow),
                      float(intensity), 4.0, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    # the flash: a blinding full-frame white pulse in the first instant
    if time < 1.4:
        f = (1.4 - time) / 1.4
        hdr = hdr + np.array([1.0, 0.96, 0.88], np.float32) * (f * f * 3.0)

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.55, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)
