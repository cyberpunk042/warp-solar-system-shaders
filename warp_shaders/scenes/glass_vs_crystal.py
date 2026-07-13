"""Glass vs crystal — the same atoms, two kinds of order.

A **crystal** is long-range ordered — atoms on a repeating lattice. A **glass** is an
amorphous solid — frozen liquid disorder, no long-range order. Same chemistry, utterly
different arrangement: cool a liquid slowly and it crystallises; cool it fast and it
vitrifies. Here the ordered lattice (left) sits beside the amorphous glass (right). See
``docs/research/31-states-of-matter.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..scene import Scene

_G = 0.24


@wp.func
def _atom(ci: float, cj: float, glass: float) -> wp.vec2:
    x = ci * _G + 0.5 * _G * wp.mod(cj, 2.0)
    y = cj * _G * 0.87
    if glass > 0.5:
        x = x + (hash21(wp.vec2(ci * 1.3 + 5.0, cj * 2.7)) - 0.5) * _G * 0.85
        y = y + (hash21(wp.vec2(ci * 3.1, cj * 1.9 + 2.0)) - 0.5) * _G * 0.85
    return wp.vec2(x, y)


@wp.func
def _segd(p: wp.vec2, a: wp.vec2, b: wp.vec2) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.kernel
def gc_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.02, 0.025, 0.035)

    glass = float(0.0)
    if x > 0.0:
        glass = 1.0
    cj0 = wp.floor(y / (_G * 0.87) + 0.5)
    bond = float(1e9)
    atom = float(1e9)
    for dj in range(-1, 2):
        cj = cj0 + float(dj)
        ci0 = wp.floor((x - 0.5 * _G * wp.mod(cj, 2.0)) / _G + 0.5)
        for di in range(-1, 2):
            ci = ci0 + float(di)
            a = _atom(ci, cj, glass)
            atom = wp.min(atom, wp.length(p - a))
            bond = wp.min(bond, _segd(p, a, _atom(ci + 1.0, cj, glass)))
            bond = wp.min(bond, _segd(p, a, _atom(ci, cj + 1.0, glass)))

    col = col + wp.vec3(0.35, 0.5, 0.7) * wp.exp(-(bond / 0.012) ** 2.0) * 0.5   # bonds
    ag = wp.exp(-(atom / 0.055) ** 2.0)
    col = col + wp.vec3(0.95, 0.75, 0.4) * ag + wp.vec3(1.0, 0.95, 0.85) * wp.exp(-(atom / 0.02) ** 2.0) * 0.8

    # divider between the two panels
    col = col + wp.vec3(0.5, 0.7, 1.0) * wp.exp(-(x / 0.008) ** 2.0) * 0.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(gc_kernel, dim=(height, width),
              inputs=[img, float(width / height), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="glass_vs_crystal",
    description="Glass vs crystal — the same atoms in two orders: an ordered repeating "
                "lattice (left) beside an amorphous glass with no long-range order "
                "(right). Slow cooling crystallises; fast cooling vitrifies.",
    renderer=_render,
)
