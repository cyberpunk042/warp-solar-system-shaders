"""PCB — a printed circuit board with copper traces, pads and plated vias.

The board itself is fibreglass-reinforced epoxy (FR-4) clad with etched copper:
the gold lines are traces carrying signals, the ringed holes are vias that jump a
signal between layers, and the pad rows are footprints where chips solder down.
Green solder-mask coats the bare board; the exposed copper is tinned gold. This
is the substrate everything else in the round plugs into — here it tilts under a
studio light, its routing catching the glare. See
``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 45.0
_BX = 2.6
_BY = 0.12
_BZ = 1.9


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.3
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.68
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _board(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(_BX, _BY, _BZ)) - 0.02


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    board = _board(q)
    floor = p[1] + 1.4
    return wp.min(board, floor)


@wp.func
def _local(p: wp.vec3, time: float) -> wp.vec3:
    return _rot(p, time)


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(4):
        hr = 0.03 + 0.12 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.func
def _hline(lx: float, lz: float, z0: float, x0: float, x1: float, w: float) -> float:
    if lz > z0 - w and lz < z0 + w and lx > x0 and lx < x1:
        return 1.0
    return 0.0


@wp.func
def _vline(lx: float, lz: float, x0: float, z0: float, z1: float, w: float) -> float:
    if lx > x0 - w and lx < x0 + w and lz > z0 and lz < z1:
        return 1.0
    return 0.0


@wp.func
def _copper(lx: float, lz: float) -> float:
    """1 = exposed copper trace/pad, 0 = bare mask, 2 = via hole (signalled >1.5)."""
    c = float(0.0)
    w = 0.035
    c = wp.max(c, _hline(lx, lz, -1.2, -2.2, 1.8, w))
    c = wp.max(c, _hline(lx, lz, -0.6, -1.6, 2.2, w))
    c = wp.max(c, _hline(lx, lz, 0.1, -2.2, 1.2, w))
    c = wp.max(c, _hline(lx, lz, 0.7, -0.8, 2.2, w))
    c = wp.max(c, _hline(lx, lz, 1.25, -2.2, 2.0, w))
    c = wp.max(c, _vline(lx, lz, -1.6, -1.2, 1.25, w))
    c = wp.max(c, _vline(lx, lz, -0.4, -1.2, 0.7, w))
    c = wp.max(c, _vline(lx, lz, 0.8, -0.6, 1.25, w))
    c = wp.max(c, _vline(lx, lz, 1.8, -1.2, 1.25, w))
    # via grid: plated rings with a hole
    vx = lx / 0.6 - wp.floor(lx / 0.6) - 0.5
    vz = lz / 0.6 - wp.floor(lz / 0.6) - 0.5
    r = wp.sqrt(vx * vx + vz * vz) * 0.6
    if r < 0.13:
        c = 1.0                 # copper annulus
        if r < 0.06:
            return 2.0          # drilled hole
    return c


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
    for _ in range(170):
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

    if p[1] < -1.37:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    lp = _local(p, time)
    on_top = lp[1] > (_BY - 0.02)
    if on_top:
        cu = _copper(lp[0], lp[2])
        if cu > 1.5:
            img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.4   # via hole
        elif cu > 0.5:
            img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))          # gold copper
        else:
            img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))          # green mask
    else:
        img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    az = 0.8 + float(mouse[0]) * 0.01
    el = 0.62 + float(mouse[1]) * 0.005
    dist = 7.0
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.4,
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
    return ec.finish(img.numpy(), width, height, threshold=1.6)


SCENE = Scene(
    name="pcb",
    description="a printed circuit board — green solder-mask over FR-4 fibreglass, "
                "etched with gold copper traces, a grid of plated vias, and pad rows "
                "where chips solder down. The substrate the whole board is built on.",
    renderer=_render,
)
