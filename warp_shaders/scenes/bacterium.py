"""A bacterium — a simple cell that swims.

A bacterium is a full cell without a nucleus: its **DNA** floats loose and coiled in the
cytoplasm (the nucleoid), dotted with **ribosomes**, and it often swims with a rotary
helical **flagellum** driven by a molecular motor in the cell wall. See
``docs/research/33-the-cell-up-close.md``. --frames spins the flagellum.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..scene import Scene


@wp.func
def _segd(p: wp.vec2, a: wp.vec2, b: wp.vec2) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.kernel
def bact_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.02, 0.035, 0.03)

    # rod-shaped cell body (a horizontal capsule)
    body = _segd(p, wp.vec2(-0.55, 0.0), wp.vec2(0.55, 0.0)) - 0.34
    if body < 0.0:
        col = wp.vec3(0.14, 0.3, 0.18) * (0.7 + 0.4 * hash21(wp.vec2(x * 20.0, y * 20.0)))
        # coiled nucleoid DNA (two anti-phase strands)
        yd1 = 0.14 * wp.sin(x * 17.0 + 0.4)
        yd2 = 0.14 * wp.sin(x * 17.0 + 3.5)
        dna = wp.min(wp.abs(y - yd1), wp.abs(y - yd2))
        col = col + wp.vec3(0.4, 0.7, 1.0) * wp.exp(-(dna / 0.02) ** 2.0) * 0.8
        # ribosomes
        rb = hash21(wp.vec2(wp.floor(x * 16.0), wp.floor(y * 16.0)))
        if rb > 0.9:
            fx = x * 16.0 - wp.floor(x * 16.0) - 0.5
            fy = y * 16.0 - wp.floor(y * 16.0) - 0.5
            col = col + wp.vec3(1.0, 0.85, 0.5) * wp.exp(-(fx * fx + fy * fy) * 12.0) * 0.7
    # membrane rim
    col = col + wp.vec3(0.5, 0.9, 0.6) * wp.exp(-(body / 0.02) ** 2.0) * 0.7

    # helical flagellum trailing from the left pole (rotating)
    if x < -0.5:
        yt = 0.16 * wp.sin((x + 0.5) * 12.0 - time * 9.0)
        taper = wp.smoothstep(-1.8, -0.5, x) * wp.smoothstep(0.0, -0.6, x)
        dist = wp.abs(y - yt)
        col = col + wp.vec3(0.6, 0.95, 0.7) * wp.exp(-(dist / 0.02) ** 2.0) * taper * 1.2

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bact_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="bacterium",
    description="A bacterium — a rod-shaped cell with its coiled loose DNA (nucleoid) and "
                "ribosomes in the cytoplasm, swimming with a rotating helical flagellum. "
                "--frames spins the flagellum.",
    renderer=_render,
)
