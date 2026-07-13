"""The Mariana Trench — a descent into the deepest place on Earth.

The Challenger Deep reaches ~11 000 m — deeper than Everest is tall — at hundreds of
atmospheres and near-total darkness. We look down between steep trench walls that
converge into blackness, **marine snow** (falling organic detritus) drifting past, a
faint bioluminescent glimmer far below. See ``docs/research/28-the-deep-ocean.md``.
--frames falls the snow.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def trench_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                  width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    v = (float(i) + 0.5) / float(height)                 # 0 top .. 1 bottom (descent)

    # darkness deepens with descent
    depth = wp.pow(v, 0.8)
    col = wp.vec3(0.02, 0.06, 0.10) * (1.0 - 0.85 * depth)

    # converging trench walls (gap narrows with depth)
    gap = 0.72 - 0.5 * v
    ax = wp.abs(x)
    if ax > gap:
        edge = wp.smoothstep(gap, gap + 0.05, ax)
        rock = fbm3(wp.vec3(x * 3.0, v * 6.0, 0.0), 4)
        rr = fbm3(wp.vec3(x * 12.0, v * 20.0, 5.0), 3)
        wall = wp.vec3(0.06, 0.07, 0.09) * (0.35 + 0.7 * rock) * (0.3 + 0.5 * rr)
        wall = wall * (1.0 - 0.8 * depth)
        col = col * (1.0 - edge) + wall * edge

    # faint bioluminescent glow deep at the bottom
    glow = wp.exp(-(x * x) * 2.5) * wp.smoothstep(0.55, 1.0, v)
    col = col + wp.vec3(0.05, 0.25, 0.28) * glow * 0.7

    # marine snow: layers of slowly falling detritus
    snow = float(0.0)
    for layer in range(3):
        sc = 26.0 + float(layer) * 16.0
        xx = x * sc * 0.6 + float(layer) * 5.1
        yy = v * sc + time * (0.25 + 0.12 * float(layer))
        cx = wp.floor(xx)
        cy = wp.floor(yy)
        h = hash21(wp.vec2(cx, cy))
        if h > 0.978:
            fx = xx - cx - 0.5
            fy = yy - cy - 0.5
            snow = snow + wp.exp(-(fx * fx + fy * fy) * 45.0) * (0.7 / (1.0 + float(layer)))
    col = col + wp.vec3(0.7, 0.8, 0.9) * snow

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(trench_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=0.8, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="mariana_trench",
    description="A descent into the Mariana Trench — steep dark walls converging into "
                "blackness, marine snow drifting down through crushing pressure, a faint "
                "bioluminescent glimmer far below. --frames falls the snow.",
    renderer=_render,
)
