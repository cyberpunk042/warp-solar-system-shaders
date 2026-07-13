"""Capacitor — an electrolytic can beside a ceramic disc.

A capacitor stores charge on two plates separated by an insulator. The big
aluminium can is an electrolytic: a long strip of oxidised foil rolled up in
conductive gel — huge capacitance, but polarised (note the stripe marking the
minus lead and the scored vent cross on top that lets it fail safely). The little
tan disc is a ceramic: tiny, unpolarised, the workhorse decoupling part. Both
float over a studio backdrop. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import op_union, sd_cylinder, sd_ellipsoid
from ..scene import Scene

_MAXD = 40.0
# electrolytic can
_CX = -0.95
_CR = 0.62
_CH = 1.05
# ceramic disc
_DX = 1.35
_DR = 0.5


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _can_body(q: wp.vec3) -> float:
    return sd_cylinder(q - wp.vec3(_CX, 0.0, 0.0), _CH, _CR) - 0.03


@wp.func
def _can_top(q: wp.vec3) -> float:
    return sd_cylinder(q - wp.vec3(_CX, _CH, 0.0), 0.06, _CR - 0.02)


@wp.func
def _leads(q: wp.vec3) -> float:
    l0 = sd_cylinder(q - wp.vec3(_CX - 0.2, -1.6, 0.0), 0.7, 0.05)
    l1 = sd_cylinder(q - wp.vec3(_CX + 0.2, -1.6, 0.0), 0.7, 0.05)
    l2 = sd_cylinder(q - wp.vec3(_DX - 0.2, -1.35, 0.0), 0.95, 0.045)
    l3 = sd_cylinder(q - wp.vec3(_DX + 0.2, -1.35, 0.0), 0.95, 0.045)
    return wp.min(wp.min(l0, l1), wp.min(l2, l3))


@wp.func
def _disc(q: wp.vec3) -> float:
    # flattened ellipsoid disc standing up (thin in z)
    return sd_ellipsoid(q - wp.vec3(_DX, -0.15, 0.0), wp.vec3(_DR, _DR, 0.16))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_can_body(q), _can_top(q))
    d = op_union(d, _disc(q))
    d = op_union(d, _leads(q))
    floor = p[1] + 1.55
    return wp.min(d, floor)


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
        hr = 0.02 + 0.11 * float(k)
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

    if p[1] < -1.52:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    db = _can_body(q)
    dt = _can_top(q)
    dd = _disc(q)

    if dt < 0.03 and dt <= db:
        # aluminium top with a scored vent cross
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        lx = q[0] - _CX
        cross = wp.min(wp.abs(lx), wp.abs(q[2]))
        if cross < 0.05:
            col = col * 0.45
        img[i, j] = col
    elif db < 0.03:
        # blue plastic sleeve, with a pale polarity stripe on the -x face
        base = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        base = wp.cw_mul(base, wp.vec3(0.35, 0.55, 1.4))     # tint blue
        ang = wp.atan2(q[2], q[0] - _CX)
        if wp.abs(ang - 3.14159) < 0.5 or wp.abs(ang + 3.14159) < 0.5:
            stripe = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
            base = stripe * 1.1
        img[i, j] = base
    elif dd < 0.03:
        img[i, j] = ec.lit(n, rd, 6, ao, wp.vec3(0.0, 0.0, 0.0))   # ceramic disc
    else:
        img[i, j] = ec.lit(n, rd, 3, ao, wp.vec3(0.0, 0.0, 0.0))   # leads


def _render(width, height, time, mouse, device):
    az = 0.7 + float(mouse[0]) * 0.01
    el = 0.28 + float(mouse[1]) * 0.005
    dist = 7.2
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.35,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.1, -0.1, 0.0)
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
    name="capacitor",
    description="a big electrolytic can (blue sleeve, minus-stripe, scored vent "
                "cross on top) beside a small tan ceramic disc — two ways to store "
                "charge, polarised and not. Floating over a studio backdrop, turning.",
    renderer=_render,
)
