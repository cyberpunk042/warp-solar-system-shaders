"""SRAM cell — six transistors that latch a bit: the fastest memory.

Two inverters wired output-to-input form a loop with exactly two stable states:
whichever way it tips, it holds — a 1 or a 0, forever, as long as the power is on
(no refresh, unlike DRAM). That is four transistors; two more "access" transistors
(green) let the word line connect the two storage nodes to the bit lines to read or
write. Six transistors per bit makes SRAM big and expensive, but it is the fastest
memory there is — which is why it builds a processor's caches. Shown top-down as a
cell layout; node Q glows (this cell holds a 1). See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.35
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _sub(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.32, 0.0), wp.vec3(1.75, 0.14, 1.35)) - 0.02


@wp.func
def _pmos(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(-0.55, -0.1, -0.85), wp.vec3(0.3, 0.12, 0.28)) - 0.01
    b = sd_box(q - wp.vec3(0.55, -0.1, -0.85), wp.vec3(0.3, 0.12, 0.28)) - 0.01
    return wp.min(a, b)


@wp.func
def _nmos(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(-0.55, -0.1, -0.1), wp.vec3(0.3, 0.12, 0.28)) - 0.01
    b = sd_box(q - wp.vec3(0.55, -0.1, -0.1), wp.vec3(0.3, 0.12, 0.28)) - 0.01
    return wp.min(a, b)


@wp.func
def _access(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(-1.25, -0.1, 0.55), wp.vec3(0.28, 0.12, 0.26)) - 0.01
    b = sd_box(q - wp.vec3(1.25, -0.1, 0.55), wp.vec3(0.28, 0.12, 0.26)) - 0.01
    return wp.min(a, b)


@wp.func
def _nodes(q: wp.vec3) -> float:
    # two storage nodes Q (x<0) and Qbar (x>0)
    a = sd_box(q - wp.vec3(-0.55, 0.02, 0.28), wp.vec3(0.2, 0.05, 0.2))
    b = sd_box(q - wp.vec3(0.55, 0.02, 0.28), wp.vec3(0.2, 0.05, 0.2))
    return wp.min(a, b)


@wp.func
def _lines(q: wp.vec3) -> float:
    wl = sd_box(q - wp.vec3(0.0, 0.06, 0.95), wp.vec3(1.55, 0.05, 0.06))     # word line
    bl = sd_box(q - wp.vec3(-1.55, 0.06, 0.2), wp.vec3(0.05, 0.05, 1.0))     # bit line
    blb = sd_box(q - wp.vec3(1.55, 0.06, 0.2), wp.vec3(0.05, 0.05, 1.0))     # bit line bar
    # cross-couple links (node Q <-> opposite inverter)
    x0 = sd_box(q - wp.vec3(0.0, 0.1, 0.28), wp.vec3(0.55, 0.03, 0.035))
    return wp.min(wp.min(wl, wp.min(bl, blb)), x0)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_sub(q), _pmos(q))
    d = op_union(d, _nmos(q))
    d = op_union(d, _access(q))
    d = op_union(d, _nodes(q))
    d = op_union(d, _lines(q))
    floor = p[1] + 0.55
    return wp.min(d, floor)


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0013
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.015 + 0.07 * float(k)
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
        if d < 0.0007 * t + 0.0003:
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

    if p[1] < -0.52:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dpm = _pmos(q)
    dnm = _nmos(q)
    dac = _access(q)
    dnd = _nodes(q)
    dln = _lines(q)
    dsb = _sub(q)
    mind = wp.min(wp.min(wp.min(dpm, dnm), wp.min(dac, dnd)), wp.min(dln, dsb))
    eps = 0.001
    if dnd <= mind + eps:
        # storage nodes: Q (x<0) holds a 1 -> bright cyan; Qbar (x>0) holds 0 -> dark
        if q[0] < 0.0:
            col = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))
            img[i, j] = col + wp.vec3(0.2, 0.8, 1.0) * 0.75
        else:
            img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.35
    elif dln <= mind + eps:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))          # metal lines
    elif dac <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(0.6, 1.7, 0.8))               # access (green)
    elif dpm <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(2.4, 1.15, 0.45))            # PMOS (amber)
    elif dnm <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(0.75, 1.05, 1.8))           # NMOS (blue)
    else:
        img[i, j] = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))       # substrate


def _render(width, height, time, mouse, device):
    az = 0.35 + float(mouse[0]) * 0.01
    el = 1.05 + float(mouse[1]) * 0.005
    dist = 5.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.2, 0.05)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.4)


SCENE = Scene(
    name="sram_cell",
    description="a six-transistor SRAM cell (top-down) — two cross-coupled "
                "inverters (amber PMOS + blue NMOS) latching a bit, plus two green "
                "access transistors to the bit lines. Node Q glows: it holds a 1. No "
                "refresh needed — the fastest memory, used for processor caches.",
    renderer=_render,
)
