"""p-n junction — the doped block that makes every diode and transistor.

Take one silicon crystal and dope one half with donors (n-type: spare electrons,
shown blue) and the other with acceptors (p-type: spare holes, shown orange).
Where they meet, electrons and holes recombine and leave a carrier-free
**depletion region** with a built-in electric field — the bright band down the
middle. That one-way barrier is the diode; gate it and it is the transistor. Here
the doped block turns under a studio light, majority carriers dotting each side,
the junction glowing between them. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import sd_box
from ..scene import Scene

_MAXD = 40.0
_BX = 2.3
_BY = 0.42
_BZ = 1.25
_DEPL = 0.34   # half-width of the depletion region


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    a = time * 0.35
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.28
    ct = wp.cos(tb)
    st = wp.sin(tb)
    q = wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])
    slab = sd_box(q, wp.vec3(_BX, _BY, _BZ)) - 0.03
    floor = p[1] + 1.5
    return wp.min(slab, floor)


@wp.func
def _local(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.35
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.28
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


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
    for k in range(5):
        hr = 0.02 + 0.10 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.func
def _dots(lp: wp.vec3, pitch: float, rad: float) -> float:
    """1.0 at a carrier-dot centre on a regular xz grid, 0 elsewhere."""
    cx = lp[0] / pitch - wp.floor(lp[0] / pitch) - 0.5
    cz = lp[2] / pitch - wp.floor(lp[2] / pitch) - 0.5
    r = wp.sqrt(cx * cx + cz * cz) * pitch
    return 1.0 - wp.clamp(r / rad, 0.0, 1.0)


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

    if p[1] < -1.47:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    lp = _local(p, time)
    # side of the junction: -1 n-type, +1 p-type, mid = depletion
    x = lp[0]
    side = wp.clamp(x / _DEPL, -1.0, 1.0)
    # base doped tint blended over silicon
    ntint = wp.vec3(0.16, 0.24, 0.42)      # n-type: cool blue
    ptint = wp.vec3(0.44, 0.22, 0.14)      # p-type: warm orange
    tint = ntint * (0.5 - 0.5 * side) + ptint * (0.5 + 0.5 * side)
    base = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
    base = wp.cw_mul(base, wp.vec3(0.7, 0.7, 0.7) + tint)

    on_top = lp[1] > (_BY - 0.02)
    emit = wp.vec3(0.0, 0.0, 0.0)
    if on_top:
        # depletion band down the middle: bright field glow, no carriers
        dep = 1.0 - wp.clamp(wp.abs(x) / _DEPL, 0.0, 1.0)
        emit = emit + wp.vec3(0.35, 0.55, 0.75) * (dep * dep * 0.55)
        # majority carriers as glowing dots, skipping the depletion region
        if x < -_DEPL:
            dv = _dots(lp, 0.42, 0.09)
            emit = emit + wp.vec3(0.25, 0.55, 1.0) * (dv * 0.8)   # electrons
        if x > _DEPL:
            dv = _dots(lp + wp.vec3(0.21, 0.0, 0.21), 0.42, 0.09)
            emit = emit + wp.vec3(1.0, 0.5, 0.15) * (dv * 0.8)    # holes

    img[i, j] = base + emit


def _render(width, height, time, mouse, device):
    az = 0.75 + float(mouse[0]) * 0.01
    el = 0.5 + float(mouse[1]) * 0.005
    dist = 6.8
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
    return ec.finish(img.numpy(), width, height, threshold=1.5)


SCENE = Scene(
    name="pn_junction",
    description="a doped silicon block — n-type (blue electrons) meeting p-type "
                "(orange holes) across the glowing depletion region with its built-in "
                "field. The one-way barrier at the heart of every diode and transistor.",
    renderer=_render,
)
