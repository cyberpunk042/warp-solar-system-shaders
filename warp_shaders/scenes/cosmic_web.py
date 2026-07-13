"""The cosmic web — the largest-scale structure of the universe.

Matter is not spread evenly: dark-matter **filaments** thread between dense
**nodes** (galaxy clusters), bounding vast near-empty **voids** — a foam. Rendered
as a ray-marched **Worley (cellular) edge** field: the Voronoi cell *edges* are the
filaments, brightening to knots where cells meet; the cell *interiors* are the
voids. Over a deep starfield. See
``docs/research/23-origin-and-large-scale-universe.md``.
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
def web_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sharp: float,
               time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    col = stars(rd) * 0.6
    g = _rs(ro, rd, 5.0)
    if g[1] < 0.0 or g[0] > 1.0e28:
        img[i, j] = col
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 72.0
    t = t0 + dt * 0.5
    acc = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    for _ in range(72):
        p = ro + rd * t
        w = worley3_f2(p * 0.42 + wp.vec3(2.0, 5.0, 1.0))
        edge = w[1] - w[0]                             # small on Voronoi edges (filaments)
        fil = wp.exp(-edge * sharp)                    # ~1 on a filament, ~0 in a void
        gasfade = 0.45 + 0.55 * fbm3(p * 2.2 + wp.vec3(0.0, time * 0.1, 0.0), 3)
        dens = fil * fil * gasfade
        ecol = wp.vec3(0.4, 0.55, 1.0) * fil + wp.vec3(1.0, 0.82, 0.5) * (fil * fil * fil * 0.5)
        acc = acc + ecol * (dens * trans * dt * 2.0)
        trans = trans * wp.exp(-dens * dt * 3.6)
        if trans < 0.02:
            t = 1.0e30
        t += dt
    img[i, j] = col + acc


def _render(width, height, time, mouse, device, period=14.0):
    prog = (time % period) / period
    sharp = 16.0 + 24.0 * prog                         # filaments sharpen over time
    az = 0.3 + time * 0.05 + float(mouse[0]) * 0.01
    el = 0.35 + float(mouse[1]) * 0.01
    dist = 8.5
    eye = (dist * math.cos(el) * math.sin(az), dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=52.0, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(web_kernel, dim=(height, width),
              inputs=[img, cam, float(sharp), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="cosmic_web",
    description="The cosmic web — dark-matter filaments threading between bright "
                "cluster nodes around vast voids, ray-marched from a Worley-edge "
                "field over a starfield. --frames sharpens the web.",
    renderer=_render,
)
