"""DRAM cell — one transistor, one capacitor: the bit inside RAM.

This is the smallest unit of a computer's main memory. A single access transistor
(gated by the word line) connects a tiny storage capacitor to the bit line. Charge
the capacitor and the cell holds a 1; drain it and it holds a 0 — billions of these
in a grid make a RAM chip. The charge leaks away in milliseconds, so the chip must
*refresh* every cell constantly (that is the "dynamic" in DRAM). The capacitor here
glows blue: this cell is storing a 1. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.42
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _sub(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.5, 0.0), wp.vec3(1.9, 0.16, 1.0)) - 0.02


@wp.func
def _sd_pads(q: wp.vec3) -> float:
    # source (under bitline) + drain (toward capacitor)
    s = sd_box(q - wp.vec3(-1.25, -0.28, 0.0), wp.vec3(0.24, 0.08, 0.32))
    d = sd_box(q - wp.vec3(-0.55, -0.28, 0.0), wp.vec3(0.24, 0.08, 0.32))
    return wp.min(s, d)


@wp.func
def _gate(q: wp.vec3) -> float:
    # poly gate bar across the channel (this is the word line contact)
    return sd_box(q - wp.vec3(-0.9, -0.12, 0.0), wp.vec3(0.1, 0.14, 0.42))


@wp.func
def _cap(q: wp.vec3) -> float:
    # storage capacitor: a tall stacked cylinder
    return sd_cylinder(q - wp.vec3(0.75, 0.05, 0.0), 0.62, 0.34) - 0.03


@wp.func
def _wires(q: wp.vec3) -> float:
    # word line over the gate (across z), bit line to the source, strap drain->cap
    wl = sd_box(q - wp.vec3(-0.9, 0.12, 0.0), wp.vec3(0.06, 0.05, 1.15))
    bl = sd_box(q - wp.vec3(-1.25, 0.1, 0.55), wp.vec3(0.9, 0.05, 0.06))
    strap = sd_box(q - wp.vec3(0.1, -0.2, 0.0), wp.vec3(0.55, 0.05, 0.07))
    return wp.min(wp.min(wl, bl), strap)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_sub(q), _sd_pads(q))
    d = op_union(d, _gate(q))
    d = op_union(d, _cap(q))
    d = op_union(d, _wires(q))
    floor = p[1] + 0.75
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
        hr = 0.015 + 0.08 * float(k)
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

    if p[1] < -0.72:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dsub = _sub(q)
    dpad = _sd_pads(q)
    dg = _gate(q)
    dc = _cap(q)
    dw = _wires(q)
    mind = wp.min(wp.min(wp.min(dsub, dpad), wp.min(dg, dc)), dw)
    eps = 0.001
    if dc <= mind + eps:
        # capacitor: aluminium can with a strong blue interior glow (storing a 1)
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        gl = wp.clamp(1.0 - wp.max(wp.dot(n, -rd), 0.0), 0.0, 1.0)
        img[i, j] = col + wp.vec3(0.15, 0.45, 1.0) * (0.35 + 0.6 * gl)
    elif dw <= mind + eps:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))       # copper wires
    elif dg <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(2.6, 1.2, 1.0))             # poly gate (reddish)
    elif dpad <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(0.8, 1.1, 1.6))            # doped n+ pads
    else:
        img[i, j] = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))      # silicon substrate


def _render(width, height, time, mouse, device):
    az = 0.62 + float(mouse[0]) * 0.01
    el = 0.44 + float(mouse[1]) * 0.005
    dist = 5.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.35,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.4)


SCENE = Scene(
    name="dram_cell",
    description="one transistor and one capacitor — the bit inside RAM. The access "
                "transistor (poly gate under the word line) connects the storage "
                "capacitor to the bit line; the capacitor glows blue, storing a 1. "
                "Billions in a grid make a memory chip.",
    renderer=_render,
)
