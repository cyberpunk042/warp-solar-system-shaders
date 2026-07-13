"""RAM stick — a DDR memory module (DIMM): DRAM chips on a stick.

Main memory ships as a module: a long thin PCB carrying a row of DRAM chips (each
one a grid of the 1T1C cells from the components round), with a gold edge connector
along the bottom that plugs into a motherboard slot. A key notch stops you inserting
it backwards or into the wrong DDR generation's slot. This is the *block* the earlier
`dram_cell` is tiled into, billions of bits per chip, eight-plus chips per stick. It
turns here over a studio backdrop. See ``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 45.0
_BX = 2.55       # board half-length
_BY = 0.58       # board half-height
_BZ = 0.055      # board half-thickness


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    # gentle wobble that keeps the broad chip face toward the camera (never edge-on)
    a = 0.32 + 0.18 * wp.sin(time * 0.6)
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.14
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    board = sd_box(q, wp.vec3(_BX, _BY, _BZ)) - 0.015
    # key notch: a slot cut up from the bottom edge, off-centre
    notch = sd_box(q - wp.vec3(0.35, -_BY, 0.0), wp.vec3(0.09, 0.2, 0.3))
    return op_subtract(board, notch)


@wp.func
def _chips(q: wp.vec3) -> float:
    # eight surface-mount DRAM packages in a row on the front face
    idx = wp.clamp(wp.floor(q[0] / 0.58 + 0.5), -4.0, 3.0)
    cx = 0.58 * idx
    return sd_box(q - wp.vec3(cx, 0.16, _BZ + 0.05),
                  wp.vec3(0.24, 0.34, 0.05)) - 0.01


@wp.func
def _label(q: wp.vec3) -> float:
    # a small label sticker high on the left
    return sd_box(q - wp.vec3(-1.4, 0.46, _BZ + 0.005), wp.vec3(0.7, 0.09, 0.02))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pcb(q), _chips(q))
    d = op_union(d, _label(q))
    return d


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0014
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.02 + 0.09 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(190):
        p = eye + rd * t
        d = _map(p, time)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.9
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, time)
    ao = _ao(p, n, time)

    if p[1] < -0.92:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dchip = _chips(q)
    dlabel = _label(q)
    dpcb = _pcb(q)
    if dchip <= dpcb and dchip <= dlabel:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))     # black DRAM chip
    elif dlabel <= dpcb:
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = col * 1.15                                        # label sticker
    else:
        # PCB: green board; gold edge fingers along the bottom band (front face)
        if q[1] < -0.42 and q[2] > (_BZ - 0.02):
            fx = q[0] / 0.11 - wp.floor(q[0] / 0.11)
            if fx > 0.25:
                img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))   # gold finger
            else:
                img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))
        else:
            img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))       # green mask


def _render(width, height, time, mouse, device):
    az = 0.0 + float(mouse[0]) * 0.01
    el = 0.32 + float(mouse[1]) * 0.005
    dist = 7.4
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.3,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.05, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.9, strength=0.28)


SCENE = Scene(
    name="ram_stick",
    description="a DDR memory module (DIMM) — a row of black DRAM chips on a green "
                "PCB stick with a gold edge connector and a key notch. The block the "
                "1T1C DRAM cell is tiled into, billions of bits per chip. Turning over "
                "a studio backdrop.",
    renderer=_render,
)
