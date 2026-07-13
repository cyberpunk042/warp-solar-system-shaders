"""GPU flagship — the advanced board dressed in a premium cooler.

The `gpu_board` hardware, now with the cover it ships in: a machined brushed-metal
shroud with chamfered edges (not cheap plastic), two axial fans set in polished
metal rings, an RGB light bar down the spine, a full-metal display bracket, and the
12VHPWR power connector. This is the "Founders"-style premium finish over the same
dense board — cosmetics on top of real hardware. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder, sd_round_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 60.0
_F0 = -1.55
_F1 = 1.4
_FR = 1.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.36 + 0.07 * wp.sin(time * 0.3)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _fanc(q: wp.vec3) -> float:
    if q[0] < -0.1:
        return _F0
    return _F1


@wp.func
def _pcb(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.16, 0.06), wp.vec3(3.55, 0.05, 1.22)) - 0.01


@wp.func
def _shroud(q: wp.vec3) -> float:
    body = sd_round_box(q - wp.vec3(0.0, 0.28, 0.0), wp.vec3(3.5, 0.32, 1.16), 0.08)
    fc = _fanc(q)
    well = sd_cylinder(q - wp.vec3(fc, 0.64, 0.0), 0.12, _FR)
    body = op_subtract(body, well)
    # polished ring lip around each fan (a shallow raised torus-like rim -> ring wall)
    return body


@wp.func
def _ring(q: wp.vec3) -> float:
    fc = _fanc(q)
    r = wp.length(wp.vec2(q[0] - fc, q[2]))
    ringwall = wp.abs(r - (_FR + 0.03)) - 0.05
    slab = wp.abs(q[1] - 0.6) - 0.06
    return wp.max(ringwall, slab)


@wp.func
def _hub(q: wp.vec3) -> float:
    fc = _fanc(q)
    return sd_cylinder(q - wp.vec3(fc, 0.52, 0.0), 0.12, 0.2)


@wp.func
def _bracket(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(-3.66, 0.24, 0.0), wp.vec3(0.05, 0.62, 1.2)) - 0.01


@wp.func
def _power(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(2.7, 0.66, -0.85), wp.vec3(0.5, 0.15, 0.22)) - 0.01


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pcb(q), _shroud(q))
    d = op_union(d, _ring(q))
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
    dsh = _shroud(q)
    drg = _ring(q)
    dhb = _hub(q)
    dbr = _bracket(q)
    dpw = _power(q)
    dpc = _pcb(q)
    mind = wp.min(wp.min(wp.min(dsh, drg), wp.min(dhb, dbr)), wp.min(dpw, dpc))
    eps = 0.0012
    if drg <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))         # polished fan ring
    elif dhb <= mind + eps:
        fc = _fanc(q)
        s = ec.fan_shade(q[0] - fc, q[2], _FR, time, 3.0)
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0)) * (0.35 + 0.7 * s)
    elif dbr <= mind + eps:
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        sl = q[2] / 0.42 - wp.floor(q[2] / 0.42)
        if sl < 0.5 and q[1] < 0.4:
            col = col * 0.25                                            # display ports
        img[i, j] = col
    elif dpw <= mind + eps:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))        # 12VHPWR
    elif dsh <= mind + eps:
        fc = _fanc(q)
        r = wp.length(wp.vec2(q[0] - fc, q[2]))
        if q[1] > 0.5 and n[1] > 0.55 and r < _FR:
            s = ec.fan_shade(q[0] - fc, q[2], _FR, time, 3.0)
            img[i, j] = wp.vec3(s, s, s * 1.08) * (0.32 + 0.4 * ao)      # fan blades in the well
        else:
            base = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.62   # brushed-metal shroud
            if wp.abs(q[2]) < 0.05 and q[1] > 0.5:
                base = base + wp.vec3(0.2, 0.55, 1.0) * 0.6              # RGB spine
            img[i, j] = base
    else:
        img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.4   # PCB edge


def _render(width, height, time, mouse, device):
    az = 0.5 + float(mouse[0]) * 0.01
    el = 0.44 + float(mouse[1]) * 0.005
    dist = 9.6
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
    name="gpu_flagship",
    description="a flagship graphics card — the advanced board under a premium "
                "brushed-metal shroud with chamfered edges, two axial fans in polished "
                "metal rings, an RGB spine, a full-metal display bracket, and a 12VHPWR "
                "connector. The cosmetic cover over real hardware.",
    renderer=_render,
)
