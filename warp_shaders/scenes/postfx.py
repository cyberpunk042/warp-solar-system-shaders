"""Post-FX + engine-primitives showcase.

A single scene that exercises the engine's newer building blocks together: a row
of **blackbody-coloured** emissive orbs (`engine.color.kelvin_to_rgb`) hit with
the shared **ray-sphere** test (`engine.intersect.ray_sphere`) over the reusable
**starfield + milky-way** background (`engine.sky`), then run through the full
host **post** chain — auto-exposure → bloom → ACES tonemap → chromatic
aberration → sharpen → vignette → film grain (`engine.post`).

    python render.py --scene postfx -o postfx.png
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.intersect import ray_sphere
from ..engine.sky import milky_way, starfield
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..scene import Scene

_N = wp.constant(5)


@wp.func
def _orb_temp(k: int) -> float:
    # five orbs from a cool red dwarf to a hot blue giant
    return 2500.0 + float(k) * 3200.0


@wp.kernel
def _postfx_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                   width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)

    col = starfield(rd) + milky_way(rd, wp.vec3(0.15, 1.0, 0.25), 0.22)

    best = float(1.0e30)
    R = float(0.7)
    for k in range(_N):
        cx = (float(k) - float(_N - 1) * 0.5) * 3.1
        center = wp.vec3(cx, 0.3 * wp.sin(time + float(k)), 0.0)
        h = ray_sphere(ro, rd, center, R)
        base = kelvin_to_rgb(_orb_temp(k))
        if h[1] > h[0] and h[0] > 0.0 and h[0] < best:
            best = h[0]
            p = ro + rd * h[0]
            n = wp.normalize(p - center)
            rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
            col = base * (0.85 + 1.2 * rim)              # coloured orb + hot rim
        else:
            # additive coloured glow halo (closest approach to the orb centre)
            oc = ro - center
            tca = -wp.dot(oc, rd)
            if tca > 0.0:
                d = wp.length(oc + rd * tca) / R
                if d > 1.0:
                    g = wp.pow(1.0 / d, 3.2) * wp.smoothstep(3.2, 1.0, d)
                    col = col + base * (g * 0.7)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    eye = (0.0, 1.6, 12.0)
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=48.0, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_postfx_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.25, strength=0.35, radius=r, passes=4)
    out = post.tonemap(hdr, mode="aces", exposure=1.05)
    out = post.chromatic_aberration(out, amount=0.004)
    out = post.sharpen(out, amount=0.35, radius=2)
    out = post.vignette(out, amount=0.35)
    out = post.film_grain(out, amount=0.02, seed=int(time * 60.0) % 997)
    return out


SCENE = Scene(name="postfx", renderer=_render,
              description="Engine showcase: blackbody-coloured emissive orbs "
                          "(engine.color) with the shared ray-sphere test over "
                          "the reusable starfield + milky way, run through the "
                          "full post chain (auto-exposure/bloom/ACES/CA/sharpen/"
                          "vignette/grain).")
