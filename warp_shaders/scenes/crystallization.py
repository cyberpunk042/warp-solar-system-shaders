"""Crystallization — a dendritic front snapping a melt into a lattice.

Freeze a supercooled liquid and a **crystallisation front** sweeps through it, atoms
snapping onto the lattice and releasing latent heat. It grows fastest along preferred
crystal axes, throwing out six-fold **dendritic** fingers — a snowflake forming. See
``docs/research/31-states-of-matter.md``. --frames grows the crystal.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def crystal_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, front: float,
                   time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    r = wp.length(wp.vec2(x, y))
    ang = wp.atan2(y, x)

    # supercooled liquid background
    liq = 0.5 + 0.5 * fbm3(wp.vec3(x * 3.0, y * 3.0, time * 0.1), 3)
    col = wp.vec3(0.02, 0.05, 0.12) * (0.7 + 0.5 * liq)

    # dendritic reach: fastest along six hex axes, feathery edge from noise
    spoke = wp.pow(0.5 + 0.5 * wp.cos(ang * 6.0), 4.0)
    branch = wp.pow(0.5 + 0.5 * wp.cos(ang * 6.0), 1.5) \
        * (0.5 + 0.5 * wp.cos(r * 18.0 - time))            # side branches along the arms
    feather = 0.07 * fbm3(wp.vec3(wp.cos(ang) * 5.0, wp.sin(ang) * 5.0, r * 3.0), 4)
    reach = front * (0.42 + 0.55 * spoke + 0.1 * branch * spoke) + feather
    cover = wp.smoothstep(0.05, -0.03, r - reach)

    if cover > 0.001:
        facet = fbm3(wp.vec3(x * 7.0, y * 7.0, 3.0), 4)
        ice = wp.vec3(0.55, 0.78, 1.0) * (0.45 + 0.6 * facet)
        spark = hash21(wp.vec2(wp.floor(x * 120.0), wp.floor(y * 120.0)))
        if spark > 0.992:
            ice = ice + wp.vec3(1.0, 1.0, 1.0) * 0.7
        col = col * (1.0 - cover) + ice * cover
        # bright growth front (latent-heat glow)
        edge = cover * (1.0 - cover) * 4.0
        col = col + wp.vec3(0.7, 0.9, 1.0) * edge * 0.7

    img[i, j] = col


def _render(width, height, time, mouse, device):
    front = 0.15 + 0.16 * time                        # front radius grows with time
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(crystal_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(front), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="crystallization",
    description="A crystallisation front sweeping a supercooled melt — the lattice "
                "growing fastest along six hex axes into dendritic fingers (a snowflake "
                "forming), the growth front glowing with released latent heat. "
                "--frames grows the crystal.",
    renderer=_render,
)
