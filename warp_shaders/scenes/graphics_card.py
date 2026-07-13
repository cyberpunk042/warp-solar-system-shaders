"""Graphics card — the whole assembled GPU board.

Everything the round built, put together: the GPU package and its GDDR memory sit
on a long PCB, fed by a bank of VRM power stages; a plastic shroud and a finned
heatsink cover them, cooled by two fans; a gold PCIe edge connector plugs into the
motherboard; an eight-pin connector brings in extra power; and a metal I/O bracket
carries the display outputs. This is the finished product you slot into a PC — and
the thing whose die (`gpu_floorplan`) is the target of the "virtual graphics card"
to come. See ``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 55.0
_FAN0 = -1.25
_FAN1 = 1.25
_FANR = 0.92


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.35 + 0.1 * wp.sin(time * 0.4)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.04, 0.05), wp.vec3(2.85, 0.05, 1.2)) - 0.01


@wp.func
def _shroud(q: wp.vec3) -> float:
    body = sd_box(q - wp.vec3(0.0, 0.28, 0.0), wp.vec3(2.75, 0.3, 1.12)) - 0.02
    # shallow circular wells for the two fans
    w0 = sd_cylinder(q - wp.vec3(_FAN0, 0.62, 0.0), 0.12, _FANR)
    w1 = sd_cylinder(q - wp.vec3(_FAN1, 0.62, 0.0), 0.12, _FANR)
    body = op_subtract(body, w0)
    return op_subtract(body, w1)


@wp.func
def _fanhub(q: wp.vec3) -> float:
    h0 = sd_cylinder(q - wp.vec3(_FAN0, 0.5, 0.0), 0.12, 0.2)
    h1 = sd_cylinder(q - wp.vec3(_FAN1, 0.5, 0.0), 0.12, 0.2)
    return wp.min(h0, h1)


@wp.func
def _bracket(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(-2.95, 0.2, 0.0), wp.vec3(0.05, 0.55, 1.15)) - 0.01


@wp.func
def _power(q: wp.vec3) -> float:
    # 8-pin power connector block on the +x top edge
    return sd_box(q - wp.vec3(2.5, 0.62, -0.75), wp.vec3(0.42, 0.16, 0.22)) - 0.01


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pcb(q), _shroud(q))
    d = op_union(d, _fanhub(q))
    d = op_union(d, _bracket(q))
    return op_union(d, _power(q))


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


@wp.func
def _fan_shade(q: wp.vec3, base: wp.vec3) -> wp.vec3:
    # if q is over a fan well top, paint hub + swept blades + rim
    fc = _FAN0
    if q[0] > 0.0:
        fc = _FAN1
    dx = q[0] - fc
    dz = q[2]
    r = wp.sqrt(dx * dx + dz * dz)
    if r > _FANR:
        return base
    ang = wp.atan2(dz, dx)
    blade = 0.5 + 0.5 * wp.sin(ang * 9.0 + r * 4.0)
    shade = 0.22 + 0.5 * blade
    if r < 0.2:
        shade = 0.35                       # hub
    if r > _FANR - 0.06:
        shade = 0.6                        # rim
    return wp.vec3(shade, shade, shade * 1.05)


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
    for _ in range(210):
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
    dp = _pcb(q)
    dsh = _shroud(q)
    dhub = _fanhub(q)
    dbr = _bracket(q)
    dpw = _power(q)
    mind = wp.min(wp.min(wp.min(dp, dsh), wp.min(dhub, dbr)), dpw)
    eps = 0.001
    if dbr <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))     # metal I/O bracket
    elif dpw <= mind + eps:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))     # power connector
    elif dhub <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, _fan_shade(q, wp.vec3(1.0, 1.0, 1.0)))  # fan blades
    elif dsh <= mind + eps:
        base = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        # on the top face, paint the fans into their wells
        if q[1] > 0.5 and n[1] > 0.6:
            img[i, j] = wp.cw_mul(base + wp.vec3(0.2, 0.2, 0.2), _fan_shade(q, wp.vec3(1.0, 1.0, 1.0)))
        else:
            img[i, j] = base                                          # dark shroud
    elif dp <= mind + eps:
        # PCB: green, with gold PCIe fingers along the -z bottom edge
        if q[2] < -1.0 and q[1] < -0.02:
            fx = q[0] / 0.13 - wp.floor(q[0] / 0.13)
            if fx > 0.3 and wp.abs(q[0] - 0.6) > 0.18:
                img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))
            else:
                img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))
        else:
            img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))
    else:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    az = 0.5 + float(mouse[0]) * 0.01
    el = 0.5 + float(mouse[1]) * 0.005
    dist = 8.2
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.4,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.6)


SCENE = Scene(
    name="graphics_card",
    description="a complete graphics card — GPU + GDDR on a long PCB under a plastic "
                "shroud with two fans and a finned heatsink, a gold PCIe edge connector, "
                "an 8-pin power connector, and a metal I/O bracket. The finished board.",
    renderer=_render,
)
