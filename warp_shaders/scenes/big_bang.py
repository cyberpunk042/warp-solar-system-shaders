"""The Big Bang — space itself expanding from a hot dense beginning.

A blinding flash, then a hot **plasma** filling the frame as the scale factor
grows, its blackbody temperature cooling (white → blue → gold → red → dark) while
quantum **density fluctuations** (fBm) — the seeds of all future structure —
ripple through it. Loops; animate with ``--frames``. See
``docs/research/23-origin-and-large-scale-universe.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import blackbody
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def bang_kernel(img: wp.array2d(dtype=wp.vec3), a: float, temp: float, flash: float,
                time: float, aspect: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0
    r = wp.sqrt(x * x + y * y)

    # the expanding fireball: bright inside the horizon radius `a`, fading out
    inside = wp.smoothstep(a + 0.35, a - 0.15, r)
    # density fluctuations, stretched by expansion (comoving)
    n = fbm3(wp.vec3(x / a * 2.2, y / a * 2.2, time * 0.3), 5)
    fluct = 0.5 + 1.1 * n
    # temperature: cools over time and toward the cooler outer edge
    heat = wp.clamp(temp - 0.45 * (r / wp.max(a, 0.1)), 0.0, 1.0)
    col = blackbody(heat) * (inside * fluct)
    # a bright hot core + the initial flash
    core = wp.exp(-(r * r) / (a * a) * 2.0)
    col = col + wp.vec3(1.0, 0.95, 0.9) * (core * (0.4 + flash))
    col = col + wp.vec3(0.9, 0.95, 1.0) * (flash * wp.exp(-r * r * 1.2))
    img[i, j] = col


def _render(width, height, time, mouse, device, period=9.0):
    prog = (time % period) / period
    a = 0.06 + prog * prog * 2.6                       # scale factor grows
    temp = 1.0 - 0.72 * prog                           # global cooling
    flash = 0.0
    if prog < 0.12:
        flash = (1.0 - prog / 0.12) * 3.0              # the initial blaze
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bang_kernel, dim=(height, width),
              inputs=[img, float(a), float(temp), float(flash), float(time),
                      float(width / height), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="big_bang",
    description="The Big Bang — an expanding hot plasma cooling from white to red "
                "as the scale factor grows, rippled by the density fluctuations that "
                "seed all structure. --frames animates the expansion.",
    renderer=_render,
)
