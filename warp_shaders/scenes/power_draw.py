"""Power draw — the mind revs the RTX board up, pulling energy in through its rails.

The ignition, before any overflow. The same real board from ``gpu_board`` (the
RTX 6000 Pro Blackwell), lit close-up while the mind draws power **through the actual
board**: cold-blue electron current streams in from the 12VHPWR connector through the
VRM chokes into the die, up the PCIe edge, and the die pushes it out to fill the GDDR7
ring — white photon flashes riding along. The die floorplan brightens and pulses as it
revs, faster and faster, right up to the edge of overflow (but not over). Animate with
``--frames`` to spin it up. See ``docs/research/37-gpu-singularity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import gpu_fx as fx
from ..engine import post
from ..scene import Scene
from .gpu_board import _die_top, _mem, board_map, board_shade
from .gpu_singularity import _bao, _bnormal, _flow, _gddr, _sstep

_MAXD = 60.0
_DIE = wp.constant(wp.vec3(-0.75, 0.30, 0.05))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, rev: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.05)
    hit = int(0)
    for _ in range(220):
        p = eye + rd * t
        d = board_map(p)
        if d < 0.0008 * t + 0.0005:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    t_end = _MAXD
    surf = fx.void_bg(rd, time, 0.25 + 0.4 * rev)
    if hit == 1:
        t_end = t
        p = eye + rd * t
        n = _bnormal(p)
        ao = _bao(p, n)
        base = board_shade(p, n, rd, ao, 0.0)
        if _die_top(p) < 0.02 and p[1] > 0.18:
            pulse = 0.5 + 0.5 * wp.sin(time * (5.0 + 14.0 * rev))
            base = base + wp.vec3(0.15, 0.4, 0.9) * (rev * (0.25 + 0.5 * pulse))   # die revs
        elif _mem(p) < 0.02:
            base = base + wp.vec3(0.1, 0.3, 0.7) * (rev * 0.3)                     # memory fills
        surf = base

    # electron / photon current flowing over the real board
    vol = wp.vec3(0.0, 0.0, 0.0)
    steps = 46
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    for _ in range(steps):
        pv = eye + rd * tv
        vol += _flow(pv, time, rev) * dt
        tv += dt

    img[i, j] = surf + vol * 0.85


def _render(width, height, time, mouse, device):
    rev = _sstep(0.0, 4.5, time)

    az = 0.7 + time * 0.02 + float(mouse[0]) * 0.01
    el = 0.42 + float(mouse[1]) * 0.005
    dist = 6.8
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.5,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(-0.5, 0.1, 0.1)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov, rev],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.015))
    hdr = post.bloom(hdr, threshold=1.15, strength=0.5, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="power_draw",
    description="the RTX board revving up — cold-blue electron current drawn in from the "
                "12VHPWR connector through the VRM into the die and up the PCIe edge, then "
                "out to fill the GDDR7 ring, photons flashing white. The ignition before "
                "the overflow, close on the real board. Animate with --frames.",
    renderer=_render,
)
