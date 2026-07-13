"""Structure formation — gravity growing the cosmic web from a smooth start.

A slice of the universe: a near-uniform density field at early times sharpens, as
gravity amplifies the tiny fluctuations, into **clusters** (bright knots) strung
along **filaments** bounding dark **voids** — the cosmic web assembling. Loops;
animate with ``--frames``. See
``docs/research/23-origin-and-large-scale-universe.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3, worley3_f2
from ..scene import Scene


@wp.kernel
def struct_kernel(img: wp.array2d(dtype=wp.vec3), grow: float, aspect: float,
                  time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect * 3.0
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0 * 3.0
    p = wp.vec3(x, y, 1.5)

    dens = fbm3(p * 0.9, 5)                             # the density field
    # gravitational sharpening: contrast rises with `grow`
    contrast = 1.0 + grow * 7.0
    d = wp.clamp(0.5 + (dens - 0.5) * contrast, 0.0, 1.0)
    # filaments (Voronoi edges) fade in as structure forms
    w = worley3_f2(p * 0.6 + wp.vec3(3.0, 1.0, 0.0))
    fil = wp.exp(-(w[1] - w[0]) * (14.0 + 20.0 * grow)) * grow
    field = wp.max(d, fil * 0.8)

    # colour: dark void → blue filament → warm cluster
    void = wp.vec3(0.02, 0.02, 0.05)
    fila = wp.vec3(0.2, 0.35, 0.8)
    clus = wp.vec3(1.0, 0.85, 0.55)
    col = void * (1.0 - field)
    col = col + fila * wp.smoothstep(0.4, 0.75, field)
    col = col + clus * wp.smoothstep(0.8, 1.0, field)
    img[i, j] = col


def _render(width, height, time, mouse, device, period=12.0):
    grow = (time % period) / period                    # 0 = smooth, 1 = full web
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(struct_kernel, dim=(height, width),
              inputs=[img, float(grow), float(width / height), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="structure_formation",
    description="Structure formation — a slice of the universe sharpening from a "
                "near-uniform field into clusters (warm knots) on filaments around "
                "dark voids as gravity grows the cosmic web. --frames animates.",
    renderer=_render,
)
