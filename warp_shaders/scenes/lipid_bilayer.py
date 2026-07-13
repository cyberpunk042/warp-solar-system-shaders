"""A lipid bilayer — the membrane that defines "inside".

Every cell is wrapped in a **lipid bilayer**: two sheets of phospholipids, their
water-loving round **heads** facing the water on each side and their oily twin **tails**
hidden in the middle — a self-assembling, fluid barrier two molecules thick, studded
with **proteins** (here a channel). See ``docs/research/33-the-cell-up-close.md``.
--frames ripples the membrane.
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


@wp.func
def _lipid(p: wp.vec2, hx: float, mid: float, halft: float, up: float) -> wp.vec3:
    hy = mid + up * (halft + 0.04)                 # head sits at the water side
    ty = mid + up * 0.02                            # tails reach the midplane
    col = wp.vec3(0.0, 0.0, 0.0)
    # round hydrophilic head
    dh = wp.length(p - wp.vec2(hx, hy))
    col = col + wp.vec3(1.0, 0.55, 0.28) * wp.exp(-(dh / 0.03) ** 2.0)
    # two hydrophobic tails
    d1 = _segd(p, wp.vec2(hx - 0.02, hy - up * 0.03), wp.vec2(hx - 0.012, ty))
    d2 = _segd(p, wp.vec2(hx + 0.02, hy - up * 0.03), wp.vec2(hx + 0.012, ty))
    tail = wp.min(d1, d2)
    col = col + wp.vec3(0.95, 0.85, 0.5) * wp.exp(-(tail / 0.012) ** 2.0) * 0.7
    return col


@wp.kernel
def bilayer_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                   width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)

    mid = 0.06 * wp.sin(x * 2.4 + time)
    halft = 0.3
    s = 0.088

    # water on both sides (with a few ions/water dots)
    col = wp.vec3(0.05, 0.11, 0.24)
    wcx = wp.floor(x * 11.0)
    wcy = wp.floor(y * 11.0 + time)
    wd = hash21(wp.vec2(wcx, wcy))
    wfx = x * 11.0 - wcx - 0.5
    wfy = y * 11.0 - wcy - 0.5
    wdot = wp.exp(-(wfx * wfx + wfy * wfy) * 9.0)
    if wd > 0.93 and wp.abs(y - mid) > halft + 0.06:
        col = col + wp.vec3(0.3, 0.45, 0.7) * wdot * 0.7

    # lipids (skip the region occupied by the channel protein)
    if wp.abs(x) > 0.17:
        ci = wp.floor(x / s + 0.5)
        for di in range(-1, 2):
            hx = (ci + float(di)) * s
            col = col + _lipid(p, hx, mid, halft, 1.0)
            col = col + _lipid(p, hx, mid, halft, -1.0)

    # channel protein spanning the bilayer, with an open pore
    if wp.abs(x) < 0.17 and wp.abs(y - mid) < halft + 0.1:
        wall = wp.abs(wp.abs(x) - 0.12)
        col = wp.vec3(0.45, 0.3, 0.6) + wp.vec3(0.5, 0.4, 0.7) * wp.exp(-(wall / 0.05) ** 2.0)
        if wp.abs(x) < 0.06:                       # the pore (water/ions pass)
            col = wp.vec3(0.06, 0.14, 0.3)
            if wd > 0.88:
                col = col + wp.vec3(0.4, 0.6, 0.9) * wdot * 0.8

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bilayer_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.006))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.25, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="lipid_bilayer",
    description="A lipid bilayer — two sheets of phospholipids, round hydrophilic heads "
                "facing the water and twin oily tails meeting in the middle, forming the "
                "cell membrane, with an embedded channel protein and a water-filled pore. "
                "--frames ripples the membrane.",
    renderer=_render,
)
