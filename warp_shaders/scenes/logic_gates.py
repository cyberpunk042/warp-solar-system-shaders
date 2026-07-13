"""A NAND gate — the universal logic gate.

Switches combine into **gates** that compute boolean functions. **NAND** is
*universal*: any circuit can be built from NANDs alone. Here the classic
AND-body-plus-inversion-**bubble** symbol, two input wires and one output, with
the inputs cycling 00 → 01 → 10 → 11 and the output showing NOT(A AND B):
bright (1) except when both inputs are 1. See ``docs/research/26-the-machine.md``.
--frames cycles the truth table.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


@wp.func
def _box(p: wp.vec2, c: wp.vec2, b: wp.vec2) -> float:
    d = wp.vec2(wp.abs(p[0] - c[0]) - b[0], wp.abs(p[1] - c[1]) - b[1])
    return wp.length(wp.vec2(wp.max(d[0], 0.0), wp.max(d[1], 0.0))) + wp.min(wp.max(d[0], d[1]), 0.0)


@wp.func
def _seg(p: wp.vec2, a: wp.vec2, b: wp.vec2) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.func
def _glow(d: float, w: float) -> float:
    return wp.exp(-(d / w) * (d / w))


@wp.func
def _wire(p: wp.vec2, a: wp.vec2, b: wp.vec2, bit: float, time: float,
          dir_: float) -> wp.vec3:
    d = _seg(p, a, b)
    on = wp.step(0.5 - bit)                      # 1.0 when bit >= 0.5
    base = wp.vec3(0.5, 0.9, 1.0) * on + wp.vec3(0.25, 0.1, 0.12) * (1.0 - on)
    col = base * _glow(d, 0.01) * (0.5 + 0.7 * on)
    # travelling pulse along the wire when the bit is 1
    t = wp.clamp(wp.dot(p - a, b - a) / wp.dot(b - a, b - a), 0.0, 1.0)
    ph = wp.mod(t - time * 0.5 * dir_, 0.34)
    pul = wp.pow(wp.max(1.0 - wp.abs(ph - 0.17) * 12.0, 0.0), 3.0)
    col = col + wp.vec3(0.8, 1.0, 1.0) * pul * on * _glow(d, 0.02) * 2.2
    return col


@wp.kernel
def nand_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, a: float, b: float,
                out: float, time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.02, 0.03, 0.05)

    # gate body: AND D-shape = union of a box (left) and a disk (right edge)
    body = wp.min(_box(p, wp.vec2(-0.18, 0.0), wp.vec2(0.22, 0.3)),
                  wp.length(p - wp.vec2(0.04, 0.0)) - 0.3)
    if body < 0.0:
        col = wp.vec3(0.08, 0.12, 0.18)         # faint fill
    col = col + wp.vec3(0.55, 0.8, 1.0) * _glow(wp.abs(body), 0.012) * 0.9
    # inversion bubble at the output (this is what makes it NAND, not AND)
    bub = wp.length(p - wp.vec2(0.4, 0.0)) - 0.055
    col = col + wp.vec3(1.0, 0.7, 0.4) * _glow(wp.abs(bub), 0.012) * 0.9

    # input wires (A upper, B lower) and output wire
    col = col + _wire(p, wp.vec2(-0.95, 0.15), wp.vec2(-0.4, 0.15), a, time, 1.0)
    col = col + _wire(p, wp.vec2(-0.95, -0.15), wp.vec2(-0.4, -0.15), b, time, 1.0)
    col = col + _wire(p, wp.vec2(0.455, 0.0), wp.vec2(0.98, 0.0), out, time, 1.0)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    step = int(time * 0.5) % 4
    a = float(step & 1)
    b = float((step >> 1) & 1)
    out = 1.0 - a * b                            # NAND
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(nand_kernel, dim=(height, width),
              inputs=[img, float(width / height), a, b, out, float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.45, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="logic_gates",
    description="A NAND gate — the universal logic gate (any circuit is buildable from "
                "NANDs alone). The AND D-shape body with an inversion bubble, two input "
                "wires and an output cycling the truth table: output is 1 except when "
                "both inputs are 1. --frames cycles A,B.",
    renderer=_render,
)
