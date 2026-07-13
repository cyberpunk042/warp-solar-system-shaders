"""Graphics card — a high-end triple-fan open-air GPU board.

Everything the round built, assembled into a flagship card: the GPU package and
its GDDR memory on a long PCB, a bank of VRM power stages feeding it, a thick
finned heatsink, and a plastic shroud carrying **three** axial fans. A backplate
stiffens the rear, dual 8-pin connectors bring in the power, a metal bracket
carries the display outputs, and an RGB strip lights the top edge. This is the
open-air cooler most desktop cards use — air pulled down through the fins and
spilled into the case. See ``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 60.0
_FANS = wp.constant(wp.vec3(-2.0, 0.0, 2.0))   # three fan centres in x
_FANR = 0.98
_TOP = 0.62                                     # shroud top y


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.35 + 0.08 * wp.sin(time * 0.35)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.14, 0.06), wp.vec3(3.55, 0.05, 1.18)) - 0.01


@wp.func
def _backplate(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.1, -0.24, 0.06), wp.vec3(3.4, 0.03, 1.12)) - 0.01


@wp.func
def _fan_dx(q: wp.vec3) -> float:
    # nearest fan centre distance in x (fans laid along x)
    d0 = wp.abs(q[0] - _FANS[0])
    d1 = wp.abs(q[0] - _FANS[1])
    d2 = wp.abs(q[0] - _FANS[2])
    return wp.min(wp.min(d0, d1), d2)


@wp.func
def _fanc(q: wp.vec3) -> float:
    if q[0] < -1.0:
        return _FANS[0]
    if q[0] > 1.0:
        return _FANS[2]
    return _FANS[1]


@wp.func
def _shroud(q: wp.vec3) -> float:
    body = sd_box(q - wp.vec3(0.0, 0.3, 0.0), wp.vec3(3.5, 0.32, 1.14)) - 0.03
    # three shallow fan wells in the top
    fc = _fanc(q)
    well = sd_cylinder(q - wp.vec3(fc, 0.66, 0.0), 0.1, _FANR)
    return op_subtract(body, well)


@wp.func
def _hub(q: wp.vec3) -> float:
    fc = _fanc(q)
    return sd_cylinder(q - wp.vec3(fc, 0.54, 0.0), 0.12, 0.22)


@wp.func
def _bracket(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(-3.65, 0.24, 0.0), wp.vec3(0.05, 0.6, 1.18)) - 0.01


@wp.func
def _power(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(2.6, 0.66, -0.82), wp.vec3(0.44, 0.17, 0.2)) - 0.01
    b = sd_box(q - wp.vec3(3.15, 0.66, -0.82), wp.vec3(0.44, 0.17, 0.2)) - 0.01
    return wp.min(a, b)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pcb(q), _backplate(q))
    d = op_union(d, _shroud(q))
    d = op_union(d, _hub(q))
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
    for _ in range(220):
        p = eye + rd * t
        d = _map(p, time)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.86
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
    dbp = _backplate(q)
    dsh = _shroud(q)
    dhub = _hub(q)
    dbr = _bracket(q)
    dpw = _power(q)
    mind = wp.min(wp.min(wp.min(dp, dbp), wp.min(dsh, dhub)), wp.min(dbr, dpw))
    eps = 0.0012
    if dbr <= mind + eps:
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        if wp.abs(wp.floor(q[2] / 0.4) - q[2] / 0.4 * 0.0) > -1.0:      # display port slots
            sl = q[2] / 0.4 - wp.floor(q[2] / 0.4)
            if sl < 0.55 and q[1] < 0.4:
                col = col * 0.3
        img[i, j] = col
    elif dpw <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))             # power connectors
        img[i, j] = col
    elif dhub <= mind + eps:
        fc = _fanc(q)
        s = ec.fan_shade(q[0] - fc, q[2], _FANR, time, 3.0)
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0)) * (0.4 + s)
    elif dsh <= mind + eps:
        base = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        # fans painted into the wells on the top face
        if q[1] > 0.5 and n[1] > 0.55 and _fan_dx(q) < _FANR:
            fc = _fanc(q)
            s = ec.fan_shade(q[0] - fc, q[2], _FANR, time, 3.0)
            img[i, j] = wp.vec3(s, s, s * 1.1) * (0.28 + 0.34 * ao)
        elif q[2] > 1.05 and wp.abs(n[2]) > 0.5:
            # front face: heatsink fin grille + an RGB accent stripe near the top
            fin = q[0] / 0.09 - wp.floor(q[0] / 0.09)
            g = 0.5
            if fin < 0.4:
                g = 0.18
            out = base * g
            if q[1] > 0.55:
                out = out + wp.vec3(0.2, 0.55, 1.0) * 0.5              # RGB edge glow
            img[i, j] = out
        else:
            img[i, j] = base                                           # dark shroud
    elif dbp <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.7  # metal backplate
    else:
        # PCB edge: green with a row of VRM caps/chokes along the bottom
        col = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.5 + float(mouse[0]) * 0.01
    el = 0.46 + float(mouse[1]) * 0.005
    dist = 9.5
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.5,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.12, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.7, strength=0.32)


SCENE = Scene(
    name="graphics_card",
    description="a high-end open-air graphics card — GPU + GDDR on a long PCB under "
                "a finned heatsink and a shroud with three axial fans, a backplate, "
                "dual 8-pin power, a display I/O bracket, and an RGB edge. The flagship board.",
    renderer=_render,
)
