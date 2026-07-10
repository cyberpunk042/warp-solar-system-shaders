"""Starfield — a small standalone scene, mostly here to exercise the registry
with more than one shader and show the ``sdf`` toolkit being reused.

Twinkling procedural stars on a subtle vertical gradient. Drag (mouse) pans.
"""

import warp as wp

from ..scene import Scene
from ..sdf import hash2d


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2(float(j) / res[0], float(height - 1 - i) / res[1])

    pan = mouse[0] / res[0]
    p = wp.vec2(uv[0] * 60.0 + pan * 20.0, uv[1] * 34.0)
    cell = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    gv = wp.vec2(p[0] - cell[0] - 0.5, p[1] - cell[1] - 0.5)

    h = hash2d(cell)
    star = float(0.0)
    if h > 0.90:
        off = wp.vec2(hash2d(wp.vec2(cell[0] + 0.3, cell[1] + 0.3)) - 0.5,
                      hash2d(wp.vec2(cell[0] + 0.7, cell[1] + 0.7)) - 0.5)
        d = wp.length(wp.vec2(gv[0] - off[0] * 0.7, gv[1] - off[1] * 0.7))
        twinkle = 0.5 + 0.5 * wp.sin(time * (0.5 + h) + h * 6.28)
        star = wp.exp(-40.0 * d) * twinkle

    sky = wp.vec3(0.02, 0.03, 0.06) * (1.0 - uv[1] * 0.5)
    col = sky + wp.vec3(0.7, 0.8, 1.0) * star
    img[i, j] = col


SCENE = Scene(
    name="starfield",
    kernel=render_kernel,
    description="Twinkling procedural starfield on a gradient — a minimal registry demo.",
)
