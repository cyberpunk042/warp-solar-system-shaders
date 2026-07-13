"""Resistor — an axial through-hole resistor with the colour-band code.

A film of resistive material on a ceramic rod, sealed in a beige body, with a
tinned lead out each end. The painted rings are the value: reading brown-black-
red-gold here gives 10 x 100 = 1 kilo-ohm at +/-5%. The most basic circuit
element — it just turns volts into a proportional current (Ohm's law). It floats
here over a studio backdrop, slowly turning. See
``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import op_union, sd_cylinder
from ..scene import Scene

_MAXD = 40.0
_BR = 0.33      # body radius
_BH = 1.0       # body half-length (x)
_LR = 0.055     # lead radius


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.5
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.18
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _body(q: wp.vec3) -> float:
    # cylinder along x: feed (y, x, z) so the axis is x
    r = sd_cylinder(wp.vec3(q[1], q[0], q[2]), _BH, _BR)
    return r - 0.04  # rounded ends


@wp.func
def _leads(q: wp.vec3) -> float:
    return sd_cylinder(wp.vec3(q[1], q[0], q[2]), 2.2, _LR)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_body(q), _leads(q))
    floor = p[1] + 1.4
    return wp.min(d, floor)


@wp.func
def _is_body(q: wp.vec3) -> int:
    if _body(q) < 0.02:
        return 1
    return 0


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
def _band(x: float) -> wp.vec3:
    """Colour-band code brown-black-red-gold at 4 ring positions (else -1,-1,-1)."""
    w = 0.07
    if wp.abs(x + 0.55) < w:
        return wp.vec3(0.30, 0.15, 0.06)     # brown = 1
    if wp.abs(x + 0.32) < w:
        return wp.vec3(0.02, 0.02, 0.02)     # black = 0
    if wp.abs(x + 0.09) < w:
        return wp.vec3(0.72, 0.10, 0.07)     # red = x100
    if wp.abs(x - 0.55) < w:
        return wp.vec3(0.80, 0.62, 0.22)     # gold = 5%
    return wp.vec3(-1.0, -1.0, -1.0)


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

    q = _rot(p, time)
    if _is_body(q) == 1:
        col = ec.lit(n, rd, 6, ao, wp.vec3(0.0, 0.0, 0.0))   # ceramic body
        bc = _band(q[0])
        if bc[0] >= 0.0:
            # re-light with the band albedo (paint ring)
            band_lit = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
            band_lit = wp.cw_mul(band_lit, bc * 3.0)
            col = band_lit
        img[i, j] = col
    else:
        img[i, j] = ec.lit(n, rd, 3, ao, wp.vec3(0.0, 0.0, 0.0))   # tinned lead


def _render(width, height, time, mouse, device):
    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.32 + float(mouse[1]) * 0.005
    dist = 6.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.3,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.05, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(40.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="resistor",
    description="an axial through-hole resistor — a beige ceramic body banded "
                "brown-black-red-gold (1 kilo-ohm) with tinned leads, floating over a "
                "studio backdrop. The simplest circuit element, Ohm's law made solid.",
    renderer=_render,
)
