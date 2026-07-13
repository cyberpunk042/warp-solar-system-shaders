"""Boiling — a liquid tearing itself into vapour.

At the boiling point, added heat goes into breaking bonds (the **latent heat**), not
raising temperature: bubbles of vapour **nucleate** on the hot floor, grow, detach and
rise through the liquid, wobbling, and burst at the surface into steam. See
``docs/research/31-states-of-matter.md``. --frames rolls the boil.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def boil_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0

    surface = 0.45 + 0.04 * wp.sin(x * 6.0 + time * 3.0) + 0.03 * fbm3(wp.vec3(x * 4.0, time * 0.5, 0.0), 3)

    if y < surface:
        # water column, lighter toward the surface, hot glow from the floor
        depth = (surface - y) / 1.4
        col = wp.vec3(0.06, 0.22, 0.4) * (1.0 - 0.5 * depth) + wp.vec3(0.02, 0.06, 0.1)
        hot = wp.smoothstep(-0.7, -1.0, y)
        col = col + wp.vec3(1.0, 0.4, 0.12) * hot * 0.6
        # rising vapour bubbles
        for k in range(26):
            s = hash21(wp.vec2(float(k) * 1.3, 7.0))
            s2 = hash21(wp.vec2(float(k) * 2.7, 3.0))
            bx = (s * 2.0 - 1.0) * aspect * 0.92
            spd = 0.35 + 0.55 * s2
            ph = wp.mod(time * spd + s * 5.0, 1.0)
            by = -0.82 + ph * (surface + 0.82)
            rad = 0.018 + 0.05 * ph
            wob = 0.03 * wp.sin(time * 6.0 + s * 20.0)
            d = wp.length(wp.vec2(x - bx - wob, y - by))
            fade = wp.smoothstep(1.0, 0.85, ph)          # burst near the surface
            if d < rad:
                col = col + wp.vec3(0.5, 0.7, 0.85) * (1.0 - d / rad) * 0.4 * fade
            rim = wp.exp(-((d - rad * 0.9) / (rad * 0.18)) ** 2.0)
            col = col + wp.vec3(0.8, 0.92, 1.0) * rim * 0.7 * fade
            spec = wp.exp(-(wp.length(wp.vec2(x - bx + rad * 0.3, y - by - rad * 0.3)) / (rad * 0.25)) ** 2.0)
            col = col + wp.vec3(1.0, 1.0, 1.0) * spec * 0.5 * fade
    else:
        # steam rising above the surface
        steam = fbm3(wp.vec3(x * 3.0, y * 3.0 - time * 1.5, time * 0.3), 4)
        st = wp.smoothstep(0.5, 0.9, steam) * wp.smoothstep(1.4, surface, y)
        col = wp.vec3(0.03, 0.04, 0.06) + wp.vec3(0.7, 0.75, 0.8) * st * 0.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(boil_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="boiling",
    description="A rolling boil — vapour bubbles nucleating on the hot floor, growing and "
                "rising through the water, wobbling and bursting at the surface into steam "
                "(latent heat tearing liquid into gas). --frames rolls the boil.",
    renderer=_render,
)
