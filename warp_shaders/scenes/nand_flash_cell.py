"""NAND flash cell — a floating-gate transistor: the bit inside an SSD / NVMe.

Take an ordinary MOSFET and slip a second, completely isolated gate between the
channel and the control gate. Push electrons onto that floating gate (by quantum
tunnelling through a thin oxide) and they are trapped there — with no power at all,
for years. Trapped charge shifts the transistor's threshold, and that shift is the
stored bit; erasing tunnels the charge back off. Non-volatile, unlike DRAM. Pack
these in NAND strings and you get flash memory. The floating gate glows amber here:
charge is trapped, the cell is programmed. See ``docs/research/35-electronics-components.md``.
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
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.30
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _sub(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.55, 0.0), wp.vec3(1.7, 0.16, 0.95)) - 0.02


@wp.func
def _pads(q: wp.vec3) -> float:
    s = sd_box(q - wp.vec3(-0.95, -0.34, 0.0), wp.vec3(0.34, 0.08, 0.62))
    d = sd_box(q - wp.vec3(0.95, -0.34, 0.0), wp.vec3(0.34, 0.08, 0.62))
    return wp.min(s, d)


@wp.func
def _tunnel_ox(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.28, 0.0), wp.vec3(0.5, 0.03, 0.62))


@wp.func
def _float_gate(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.12, 0.0), wp.vec3(0.46, 0.12, 0.6)) - 0.01


@wp.func
def _inter_ox(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, 0.03, 0.0), wp.vec3(0.48, 0.03, 0.62))


@wp.func
def _ctrl_gate(q: wp.vec3) -> float:
    cg = sd_box(q - wp.vec3(0.0, 0.2, 0.0), wp.vec3(0.44, 0.13, 0.6)) - 0.01
    wl = sd_box(q - wp.vec3(0.0, 0.36, 0.0), wp.vec3(0.14, 0.06, 1.15))   # word line
    return wp.min(cg, wl)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_sub(q), _pads(q))
    d = op_union(d, _tunnel_ox(q))
    d = op_union(d, _float_gate(q))
    d = op_union(d, _inter_ox(q))
    d = op_union(d, _ctrl_gate(q))
    floor = p[1] + 0.8
    return wp.min(d, floor)


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0012
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

    if p[1] < -0.77:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dsub = _sub(q)
    dpad = _pads(q)
    dtox = _tunnel_ox(q)
    dfg = _float_gate(q)
    diox = _inter_ox(q)
    dcg = _ctrl_gate(q)
    mind = wp.min(wp.min(wp.min(dsub, dpad), wp.min(dtox, dfg)), wp.min(diox, dcg))
    eps = 0.001
    if dfg <= mind + eps:
        # floating gate: trapped charge — amber glow
        col = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))
        gl = wp.clamp(1.0 - wp.max(wp.dot(n, -rd), 0.0), 0.0, 1.0)
        img[i, j] = col * 0.5 + wp.vec3(1.0, 0.55, 0.12) * (0.45 + 0.7 * gl)
    elif dcg <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(2.4, 1.2, 1.1))            # control gate poly
    elif dtox <= mind + eps or diox <= mind + eps:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0)) + wp.vec3(0.06, 0.08, 0.12)
    elif dpad <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(0.8, 1.1, 1.6))           # doped pads
    else:
        img[i, j] = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))     # silicon substrate


def _render(width, height, time, mouse, device):
    az = 0.62 + float(mouse[0]) * 0.01
    el = 0.34 + float(mouse[1]) * 0.005
    dist = 5.4
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.25,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.12, 0.0)
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
    name="nand_flash_cell",
    description="a floating-gate transistor — the non-volatile bit inside an SSD. "
                "An isolated gate (glowing amber) is buried under the control gate and "
                "over a thin tunnel oxide; electrons tunnelled onto it stay trapped for "
                "years with no power, shifting the threshold. That trapped charge is the bit.",
    renderer=_render,
)
