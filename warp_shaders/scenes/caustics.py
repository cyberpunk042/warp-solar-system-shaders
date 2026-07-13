"""Caustics — the dancing net of light on a pool floor.

When a rippling water surface focuses sunlight, the concentrated bright curves are
**caustics** — the envelopes of refracted rays, brightest where rays pile up. The
shifting web of light on the bottom of a pool. Rendered with the classic layered
ray-folding caustic approximation. See ``docs/research/30-light-and-optics.md``.
--frames ripples the water.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


@wp.kernel
def caustic_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                   width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width))) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height))

    tau = 6.28318
    inten = 0.005
    px = wp.mod(x * tau * 1.4, tau) - 250.0
    py = wp.mod(y * tau * 1.4, tau) - 250.0
    ix = px
    iy = py
    c = float(1.0)
    for n in range(5):
        t = time * 0.6 * (1.0 - (3.5 / float(n + 1)))
        nix = px + wp.cos(t - ix) + wp.sin(t + iy)
        niy = py + wp.sin(t - iy) + wp.cos(t + ix)
        ix = nix
        iy = niy
        c = c + 1.0 / wp.length(wp.vec2(px / (wp.sin(ix + t) / inten),
                                        py / (wp.cos(iy + t) / inten)))
    c = c / 5.0
    c = 1.17 - wp.pow(c, 1.4)
    val = wp.pow(wp.abs(c), 8.0)

    # deep-water floor tinted blue-green, caustics as bright cool light
    floor = wp.vec3(0.02, 0.09, 0.13) * (0.7 + 0.5 * y)
    caustic = wp.vec3(0.45, 0.85, 1.0) * val
    img[i, j] = floor + caustic


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(caustic_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="caustics",
    description="Water caustics — the shifting bright net of light a rippling surface "
                "focuses onto a pool floor, the envelopes of refracted rays where they "
                "pile up. --frames ripples the water.",
    renderer=_render,
)
