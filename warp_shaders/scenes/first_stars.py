"""First stars — cosmic dawn, when the dark universe first lit up.

After the dark ages, gravity collapses the densest pockets of primordial H/He gas
into the first stars (Population III) — metal-free, so very massive and **blue-hot**.
They ignite one by one, lighting the surrounding gas. Loops; animate with
``--frames``. See ``docs/research/23-origin-and-large-scale-universe.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..procedural.hash import hash21
from ..scene import Scene


@wp.kernel
def stars_kernel(img: wp.array2d(dtype=wp.vec3), prog: float, aspect: float,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0

    # primordial neutral gas (dim, reddish) with structure
    gas = 0.5 + 0.6 * fbm3(wp.vec3(x * 1.6, y * 1.6, 7.0), 5)
    col = wp.vec3(0.14, 0.05, 0.05) * gas * 0.5

    # background faint galaxies-to-be / stars
    s = hash21(wp.vec2(wp.floor(x * 90.0), wp.floor(y * 90.0)))
    col = col + wp.vec3(0.3, 0.3, 0.4) * (wp.step(0.995 - s) * 0.6)

    # the first stars ignite one by one
    for k in range(7):
        fk = float(k)
        px = 1.5 * (hash21(wp.vec2(fk, 1.3)) - 0.5) * 2.0 * aspect
        py = 1.5 * (hash21(wp.vec2(fk, 7.7)) - 0.5) * 2.0
        tk = 0.08 + 0.11 * fk                          # ignition time
        ig = wp.clamp((prog - tk) / 0.06, 0.0, 1.0)
        flick = 0.85 + 0.15 * wp.sin(time * 6.0 + fk)
        d = wp.length(wp.vec2(x - px, y - py))
        star = wp.exp(-(d / 0.035) * (d / 0.035)) + 0.25 * wp.exp(-(d / 0.12) * (d / 0.12))
        col = col + wp.vec3(0.8, 0.9, 1.0) * (star * ig * flick * 1.4)
        # blue starlight lighting the nearby gas
        lit = wp.exp(-(d / 0.5) * (d / 0.5)) * ig
        col = col + wp.vec3(0.35, 0.5, 0.9) * (lit * gas * 0.5)

    img[i, j] = col


def _render(width, height, time, mouse, device, period=11.0):
    prog = (time % period) / period
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(stars_kernel, dim=(height, width),
              inputs=[img, float(prog), float(width / height), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.6, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="first_stars",
    description="Cosmic dawn — the first (Population III) stars igniting one by one "
                "in the dark primordial gas, blue-hot and lighting the cloud around "
                "them. --frames animates the ignitions.",
    renderer=_render,
)
