"""Noise gallery — a live proof of the procedural toolkit.

Six tiles, each a different generator sampled on an animated z-slice:
fbm · Perlin · Worley (F1) · ridged · domain-warp · curl (magnitude). Tinted per
tile for legibility. Doubles as the P1 verification scene.
"""

import warp as wp

from ..procedural import (
    curl3, domain_warp3, fbm3, ridged3, simplex3, worley3,
)
from ..procedural.hash import fract
from ..scene import Scene


@wp.func
def _tint(v: float, c: wp.vec3) -> wp.vec3:
    return c * v


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int,
                  time: float, mouse: wp.vec2):
    i, j = wp.tid()
    u = (float(j) + 0.5) / float(width)
    v = (float(height - 1 - i) + 0.5) / float(height)

    col_i = int(u * 3.0)
    row_i = int(v * 2.0)
    lu = fract(u * 3.0)
    lv = fract(v * 2.0)
    p = wp.vec3(lu * 4.0, lv * 4.0, time * 0.15)
    idx = row_i * 3 + col_i

    val = float(0.0)
    tint = wp.vec3(1.0, 1.0, 1.0)
    if idx == 0:
        val = fbm3(p, 6)
        tint = wp.vec3(0.55, 0.75, 1.0)
    elif idx == 1:
        val = simplex3(p) * 0.5 + 0.5
        tint = wp.vec3(1.0, 0.85, 0.5)
    elif idx == 2:
        val = wp.clamp(worley3(p), 0.0, 1.0)
        tint = wp.vec3(0.6, 1.0, 0.7)
    elif idx == 3:
        val = ridged3(p, 6)
        tint = wp.vec3(1.0, 0.6, 0.55)
    elif idx == 4:
        val = domain_warp3(p, 5, 1.0)
        tint = wp.vec3(0.85, 0.6, 1.0)
    else:
        c = curl3(p * 0.5)
        val = wp.clamp(wp.length(c) * 0.35, 0.0, 1.0)
        tint = wp.vec3(0.6, 0.9, 1.0)

    # thin grid lines between tiles
    gx = wp.min(lu, 1.0 - lu)
    gy = wp.min(lv, 1.0 - lv)
    line = wp.clamp(wp.min(gx, gy) * 30.0, 0.0, 1.0)

    c3 = _tint(val, tint) * (0.25 + 0.75 * line)
    img[i, j] = wp.vec3(wp.pow(wp.max(c3[0], 0.0), 0.4545),
                        wp.pow(wp.max(c3[1], 0.0), 0.4545),
                        wp.pow(wp.max(c3[2], 0.0), 0.4545))


SCENE = Scene(
    name="noise_gallery",
    kernel=render_kernel,
    description="Procedural toolkit demo: fbm/Perlin/Worley/ridged/domain-warp/curl tiles.",
)
