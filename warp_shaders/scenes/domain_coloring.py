"""Domain colouring — a complex function made visible.

A complex function f(z) maps the plane to the plane, so you can't graph it directly.
**Domain colouring** shows it by painting each point z of the plane: the **hue** is
the phase arg f(z) and the **brightness/bands** encode the magnitude |f(z)|. Zeros
appear as points where all colours meet (a full colour wheel winding once); **poles**
wind the wheel the opposite way. Here f(z) = (z³ − a) / (z² + b): three rotating
zeros and a pair of poles. See ``docs/research/27-mathematics-made-visible.md``.
--frames rotates the zeros.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


@wp.func
def _hue(h: float) -> wp.vec3:
    r = wp.clamp(wp.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = wp.clamp(2.0 - wp.abs(h * 6.0 - 2.0), 0.0, 1.0)
    b = wp.clamp(2.0 - wp.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return wp.vec3(r, g, b)


@wp.kernel
def dc_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, ar: float, ai: float,
              width: int, height: int):
    i, j = wp.tid()
    zr = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect * 2.2
    zi = (((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0) * 2.2

    # z^2, z^3
    z2r = zr * zr - zi * zi
    z2i = 2.0 * zr * zi
    z3r = z2r * zr - z2i * zi
    z3i = z2r * zi + z2i * zr
    # num = z^3 - a  ; den = z^2 + b   (b = 0.45)
    nr = z3r - ar
    ni = z3i - ai
    dr = z2r + 0.45
    di = z2i
    dd = dr * dr + di * di + 1e-9
    fr = (nr * dr + ni * di) / dd
    fi = (ni * dr - nr * di) / dd

    arg = wp.atan2(fi, fr) / (2.0 * 3.14159265) + 0.5      # [0,1]
    mag = wp.sqrt(fr * fr + fi * fi)
    col = _hue(arg)

    # magnitude → brightness (dark at zeros, bright at poles) + log contour rings
    val = 0.35 + 0.65 * (mag / (mag + 1.0))
    rings = 0.82 + 0.18 * wp.sin(wp.log(mag + 1e-6) * 6.2831 * 1.4)
    phase_lines = 0.85 + 0.15 * wp.sin(arg * 6.2831 * 12.0)
    col = col * val * rings * phase_lines
    # brighten the poles to white glow
    col = col + wp.vec3(1.0, 1.0, 1.0) * wp.clamp(mag - 6.0, 0.0, 3.0) * 0.15

    img[i, j] = col


def _render(width, height, time, mouse, device):
    ar = float(np.cos(time * 0.5))
    ai = float(np.sin(time * 0.5))
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(dc_kernel, dim=(height, width),
              inputs=[img, float(width / height), ar, ai, int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.006))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.25, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="domain_coloring",
    description="Domain colouring of f(z) = (z³−a)/(z²+b) — each point of the complex "
                "plane painted with hue = phase and bands = magnitude. Zeros are "
                "colour-wheel pinwheels winding one way, poles the other, with log "
                "contour rings. --frames rotates the three zeros.",
    renderer=_render,
)
