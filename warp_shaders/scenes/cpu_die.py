"""A CPU die — the processor as an aerial city of logic.

Seen from above, a chip is a city: rectangular **functional blocks** (ALU,
registers, cache, control) of many sizes, wired by a Manhattan grid of **buses**
with **data pulses** streaming along them over a dark silicon substrate. See
``docs/research/26-the-machine.md``. --frames animates the data flow.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def die_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
               width: int, height: int):
    i, j = wp.tid()
    x = ((float(j) + 0.5) / float(width)) * aspect * 2.0
    y = (float(height - 1 - i) + 0.5) / float(height) * 2.0

    # dark silicon substrate with a faint metallic mottle
    sub = 0.5 + 0.5 * fbm3(wp.vec3(x * 3.0, y * 3.0, 0.0), 4)
    col = wp.vec3(0.02, 0.05, 0.06) + wp.vec3(0.01, 0.03, 0.03) * sub

    # functional blocks on a coarse grid (varied sizes via merged cells)
    bs = 0.34
    cx = wp.floor(x / bs)
    cy = wp.floor(y / bs)
    h = hash21(wp.vec2(cx, cy))
    if h > 0.28:                                        # ~72% of cells hold a block
        fx = x / bs - cx
        fy = y / bs - cy
        m = 0.42 + 0.12 * hash21(wp.vec2(cx + 3.0, cy - 1.0))
        dx = wp.abs(fx - 0.5)
        dy = wp.abs(fy - 0.5)
        if dx < m and dy < m:
            tone = 0.1 + 0.16 * hash21(wp.vec2(cx * 1.7, cy * 2.3))
            bevel = 1.0 - 1.3 * wp.max(dx, dy)
            pad = wp.vec3(0.18, 0.26, 0.32) * tone * 6.0 * bevel
            # fine internal structure (rows of cells / gates)
            grid = wp.step(0.7 - 0.5 * wp.abs(wp.sin(fx * 40.0)) - 0.5 * wp.abs(wp.sin(fy * 40.0)))
            pad = pad + wp.vec3(0.15, 0.35, 0.4) * (grid * 0.15)
            col = pad

    # Manhattan bus grid with streaming data pulses
    ts = 0.075
    gx = wp.abs(wp.sin(x / ts * 3.1416))
    gy = wp.abs(wp.sin(y / ts * 3.1416))
    tracex = wp.max(1.0 - gx * 9.0, 0.0)
    tracey = wp.max(1.0 - gy * 9.0, 0.0)
    col = col + wp.vec3(0.12, 0.45, 0.55) * ((tracex + tracey) * 0.6)
    # pulses: bright dashes moving along the horizontal + vertical lines
    ph = wp.mod(x * 2.0 + time * 1.5, 1.0)
    pulseh = tracey * wp.pow(wp.max(1.0 - wp.abs(ph - 0.5) * 7.0, 0.0), 3.0)
    pv = wp.mod(y * 2.0 - time * 1.3, 1.0)
    pulsev = tracex * wp.pow(wp.max(1.0 - wp.abs(pv - 0.5) * 7.0, 0.0), 3.0)
    col = col + wp.vec3(0.7, 0.97, 1.0) * ((pulseh + pulsev) * 2.8)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(die_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.04)


SCENE = Scene(
    name="cpu_die",
    description="A CPU die from above — an aerial city of functional blocks (ALU, "
                "registers, cache) wired by a Manhattan bus grid with data pulses "
                "streaming along it. --frames animates the data flow.",
    renderer=_render,
)
