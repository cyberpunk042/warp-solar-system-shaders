"""Magnetic reconnection — field lines snap and fling plasma out.

When oppositely-directed magnetic field lines are pushed together — in the Sun's
corona, in Earth's magnetotail — they break and **reconnect** at an **X-point**,
converting stored magnetic energy into heat and shooting plasma out in **jets** (solar
flares, aurorae). The field near the null is hyperbolic, an X. See
``docs/research/32-electromagnetism-and-fields.md``. --frames drives the jets.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


@wp.kernel
def recon_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                 width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    col = wp.vec3(0.02, 0.02, 0.035)

    # hyperbolic field lines (level sets of x*y) — inflow above/below, outflow L/R
    band = wp.pow(0.5 + 0.5 * wp.cos(x * y * 22.0), 4.0)
    inflow = wp.smoothstep(0.0, 0.3, wp.abs(y) - wp.abs(x))     # 1 in the inflow lobes
    fcol = wp.vec3(0.3, 0.5, 1.0) * inflow + wp.vec3(1.0, 0.45, 0.3) * (1.0 - inflow)
    col = col + fcol * band * 0.5

    # the X separatrix (y = ±x) and the central current sheet
    sep = wp.exp(-((x * x - y * y) / 0.05) * ((x * x - y * y) / 0.05))
    col = col + wp.vec3(0.8, 0.9, 1.0) * sep * 0.5
    sheet = wp.exp(-(x / 0.04) * (x / 0.04)) * wp.smoothstep(0.75, 0.0, wp.abs(y))
    col = col + wp.vec3(1.0, 0.8, 0.5) * sheet * 0.8

    # plasma outflow jets along ±x from the X-point
    if wp.abs(y) < 0.16:
        jp = wp.mod(wp.abs(x) * 3.0 - time * 4.0, 1.0)
        blob = wp.pow(wp.max(1.0 - wp.abs(jp - 0.5) * 3.2, 0.0), 2.0)
        fade = wp.smoothstep(1.7, 0.12, wp.abs(x)) * wp.smoothstep(0.16, 0.0, wp.abs(y))
        col = col + wp.vec3(1.0, 0.7, 0.4) * blob * fade * 2.0

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(recon_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.009))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="magnetic_reconnection",
    description="Magnetic reconnection — oppositely-directed field lines meeting at an "
                "X-point (magnetic null), snapping and reconnecting across a glowing "
                "current sheet, flinging plasma out in jets (solar flares, aurorae). "
                "--frames drives the jets.",
    renderer=_render,
)
