"""Motherboard — the platform that wires all the parts together.

The motherboard is the big PCB every other board plugs into. The CPU drops into
its socket (centre) under the VRM heatsinks that feed it clean power; the RAM
sticks stand in the DIMM slots beside it; the graphics card slots into the long
PCIe connector; the NVMe SSD lies in an M.2 slot; and the chipset (its own little
heatsink) routes everything else. Copper traces on the inner layers carry the
signals between them all. Shown from above as a floorplan. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 55.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.35 + 0.1 * wp.sin(time * 0.4)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _board(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(2.8, 0.07, 2.4)) - 0.02


@wp.func
def _socket(q: wp.vec3) -> float:
    frame = sd_box(q - wp.vec3(-0.35, 0.13, -0.85), wp.vec3(0.72, 0.06, 0.72)) - 0.01
    ihs = sd_box(q - wp.vec3(-0.35, 0.2, -0.85), wp.vec3(0.5, 0.08, 0.5)) - 0.02
    return op_union(frame, ihs)


@wp.func
def _vrm(q: wp.vec3) -> float:
    # two finned VRM heatsinks flanking the socket (top edge + left)
    a = sd_box(q - wp.vec3(-0.35, 0.22, -1.95), wp.vec3(0.95, 0.22, 0.28)) - 0.01
    b = sd_box(q - wp.vec3(-1.75, 0.22, -0.85), wp.vec3(0.24, 0.22, 0.8)) - 0.01
    return wp.min(a, b)


@wp.func
def _ram(q: wp.vec3) -> float:
    d = float(1e9)
    for k in range(4):
        x = 1.35 + 0.24 * float(k)
        d = wp.min(d, sd_box(q - wp.vec3(x, 0.15, -0.45), wp.vec3(0.06, 0.16, 1.05)))
    return d


@wp.func
def _pcie(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(-0.2, 0.13, 1.05), wp.vec3(1.5, 0.11, 0.08))
    b = sd_box(q - wp.vec3(-0.5, 0.11, 1.75), wp.vec3(0.9, 0.09, 0.07))
    return wp.min(a, b)


@wp.func
def _chipset(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(1.7, 0.17, 1.5), wp.vec3(0.55, 0.11, 0.55)) - 0.02


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_board(q), _socket(q))
    d = op_union(d, _vrm(q))
    d = op_union(d, _ram(q))
    d = op_union(d, _pcie(q))
    return op_union(d, _chipset(q))


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
    for _ in range(200):
        p = eye + rd * t
        d = _map(p, time)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.88
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, time)
    ao = _ao(p, n, time)

    q = _rot(p, time)
    dsock = _socket(q)
    dvrm = _vrm(q)
    dram = _ram(q)
    dpci = _pcie(q)
    dchip = _chipset(q)
    dboard = _board(q)
    mind = wp.min(wp.min(wp.min(dsock, dvrm), wp.min(dram, dpci)), wp.min(dchip, dboard))
    eps = 0.001
    if dsock <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))     # socket + CPU metal
    elif dvrm <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))     # VRM heatsinks
    elif dchip <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))     # chipset heatsink
    elif dram <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.28:
            col = wp.cw_mul(col, wp.vec3(0.6, 1.4, 0.9))             # RAM slot latch colour
        img[i, j] = col
    elif dpci <= mind + eps:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))     # PCIe slot
    else:
        img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))     # green PCB


def _render(width, height, time, mouse, device):
    az = 0.45 + float(mouse[0]) * 0.01
    el = 0.78 + float(mouse[1]) * 0.005
    dist = 8.0
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.6)


SCENE = Scene(
    name="motherboard",
    description="a motherboard floorplan — the CPU socket and its VRM heatsinks, "
                "four RAM slots, two PCIe slots for the graphics card, and the chipset "
                "heatsink, all on one big green PCB. The platform every other board "
                "plugs into.",
    renderer=_render,
)
