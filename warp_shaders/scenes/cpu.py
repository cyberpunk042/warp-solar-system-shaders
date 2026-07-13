"""CPU — a processor: a silicon die under a metal heat spreader.

The processor is one big silicon die (millions of the CMOS logic gates from the
components round) flip-chip mounted on a green package substrate, then capped with
a nickel-plated copper **integrated heat spreader** (IHS) that both protects the die
and carries its heat up to a cooler. Underneath, an array of pads (LGA) or pins
mates with the motherboard socket; a corner triangle marks pin 1. Tiny surface-mount
capacitors ring the substrate to steady the supply. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box, sd_round_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 45.0
_SUB = 1.5       # substrate half-width


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.4 + 0.12 * wp.sin(time * 0.5)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _sub(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(_SUB, 0.08, _SUB)) - 0.01


@wp.func
def _ihs(q: wp.vec3) -> float:
    skirt = sd_box(q - wp.vec3(0.0, 0.14, 0.0), wp.vec3(1.16, 0.05, 1.16)) - 0.01
    cap = sd_round_box(q - wp.vec3(0.0, 0.24, 0.0), wp.vec3(0.94, 0.08, 0.94), 0.04)
    return op_union(skirt, cap)


@wp.func
def _caps(q: wp.vec3) -> float:
    # a ring of tiny SMD capacitors on the substrate border (front + back rows)
    xi = wp.clamp(wp.floor(q[0] / 0.3 + 0.5), -4.0, 4.0)
    cx = 0.3 * xi
    front = sd_box(q - wp.vec3(cx, 0.1, 1.3), wp.vec3(0.06, 0.03, 0.09))
    back = sd_box(q - wp.vec3(cx, 0.1, -1.3), wp.vec3(0.06, 0.03, 0.09))
    return wp.min(front, back)


@wp.func
def _tri(q: wp.vec3) -> float:
    # pin-1 marker at the -x/-z corner of the substrate top
    return sd_box(q - wp.vec3(-1.3, 0.085, -1.3), wp.vec3(0.14, 0.02, 0.14))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_sub(q), _ihs(q))
    d = op_union(d, _caps(q))
    return op_union(d, _tri(q))


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
    for _ in range(180):
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

    q = _rot(p, time)
    dihs = _ihs(q)
    dcaps = _caps(q)
    dtri = _tri(q)
    dsub = _sub(q)
    mind = wp.min(wp.min(dihs, dcaps), wp.min(dtri, dsub))
    eps = 0.001
    if dihs <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))     # metal IHS
    elif dtri <= mind + eps:
        img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))     # gold pin-1 tri
    elif dcaps <= mind + eps:
        col = ec.lit(n, rd, 6, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(1.2, 0.9, 0.7))          # tan SMD caps
    else:
        # green substrate, gold contact ring hint near the edge
        img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    az = 0.35 + float(mouse[0]) * 0.01
    el = 0.6 + float(mouse[1]) * 0.005
    dist = 6.4
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.05, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="cpu",
    description="a processor — a silicon die under a nickel-plated metal heat "
                "spreader (IHS) on a green package substrate, ringed by tiny SMD "
                "capacitors, a gold pin-1 triangle at one corner. Millions of CMOS "
                "gates under the metal cap.",
    renderer=_render,
)
