"""Plate tectonics — Earth's crust broken into drifting plates.

A globe of land and ocean whose surface is divided into **plates**; their
boundaries glow with heat — spreading **mid-ocean ridges** and volcanic arcs where
the crust is born and destroyed. The plates drift over `time`. See
``docs/research/25-earth-and-weather.md``. iMouse orbits.
"""

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, worley3_f2
from ..scene import Scene


@wp.kernel
def tect_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                drift: float, glow: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 1.0)
    if g[0] > 1.0e28 or g[0] < 0.0:
        img[i, j] = stars(rd)
        return
    n = wp.normalize(ro + rd * g[0])
    # land / ocean
    land = fbm3(n * 2.3, 5)
    is_land = wp.smoothstep(0.48, 0.56, land)
    ocean = wp.vec3(0.02, 0.12, 0.3)
    green = wp.vec3(0.2, 0.34, 0.14)
    brown = wp.vec3(0.4, 0.32, 0.2)
    surf = ocean * (1.0 - is_land) + (green * 0.6 + brown * 0.4 * fbm3(n * 6.0, 3)) * is_land
    # plate boundaries: Worley-cell edges (drifting), glowing hot
    w = worley3_f2(n * 3.4 + wp.vec3(drift, 0.0, 0.0))
    edge = w[1] - w[0]
    ridge = wp.exp(-edge * 18.0)
    hot = wp.vec3(1.0, 0.45, 0.12) * (ridge * (0.6 + 0.4 * wp.sin(edge * 40.0 - glow)))
    ndl = wp.max(wp.dot(n, sun), 0.0)
    col = surf * (0.1 + 0.95 * ndl) + hot * (0.5 + 0.8 * ridge)
    # atmosphere rim
    rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
    col = col + wp.vec3(0.3, 0.5, 0.9) * (rim * (0.3 + 0.7 * ndl))
    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.4 + time * 0.05 + float(mouse[0]) * 0.01
    el = 0.25 + float(mouse[1]) * 0.01
    eye = (2.7 * math.cos(el) * math.sin(az), 2.7 * math.sin(el),
           2.7 * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=44.0, aspect=width / height)
    sel = 0.5
    saz = az + 0.9
    sun = wp.vec3(math.cos(sel) * math.sin(saz), math.sin(sel), math.cos(sel) * math.cos(saz))
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(tect_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time * 0.05), float(time * 2.0),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=1.3, strength=0.4,
                     radius=max(2, int(width * 0.01)), passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="plate_tectonics",
    description="Plate tectonics — a globe of land and ocean divided into drifting "
                "plates whose boundaries glow with heat (spreading ridges + volcanic "
                "arcs). iMouse orbits; --frames drifts the plates.",
    renderer=_render,
)
