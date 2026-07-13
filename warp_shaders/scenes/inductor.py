"""Inductor — a toroidal ferrite core wound with copper wire.

An inductor resists *changes* in current: drive current through a coil and it
builds a magnetic field that stores energy and fights any change (Faraday /
Lenz). Winding the wire around a ring of ferrite (a magnetically soft ceramic)
concentrates that field inside the core and keeps it from leaking out — the shape
you see on every power-supply board. Copper turns wrap the dark toroid here, two
leads dropping off the ends, over a studio backdrop. See
``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_cylinder, sd_torus
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0
_R = 1.15       # major radius
_r = 0.34       # tube (ferrite) minor radius
_RW = 0.12      # copper wire radius
_N = 16.0       # turns


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.16
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _core(q: wp.vec3) -> float:
    return sd_torus(q, wp.vec2(_R, _r))


@wp.func
def _wire(q: wp.vec3) -> float:
    theta = wp.atan2(q[2], q[0])
    rho = wp.length(wp.vec2(q[0], q[2])) - _R
    phi = _N * theta
    cx = (_r + 0.03) * wp.cos(phi)
    cy = (_r + 0.03) * wp.sin(phi)
    d = wp.length(wp.vec2(rho - cx, q[1] - cy)) - _RW
    return d


@wp.func
def _leads(q: wp.vec3) -> float:
    l0 = sd_cylinder(q - wp.vec3(_R - 0.1, -1.4, 0.0), 0.9, 0.055)
    l1 = sd_cylinder(q - wp.vec3(-_R + 0.1, -1.4, 0.0), 0.9, 0.055)
    return wp.min(l0, l1)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_core(q), _wire(q))
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
        hr = 0.02 + 0.10 * float(k)
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
    for _ in range(200):
        p = eye + rd * t
        d = _map(p, time)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.85
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
    dc = _core(q)
    dw = _wire(q)
    dl = _leads(q)
    if dw <= dc and dw <= dl:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))   # copper wire
    elif dl < dc:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))   # copper leads
    else:
        # ferrite: dark magnetic ceramic
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(1.4, 1.4, 1.5))


def _render(width, height, time, mouse, device):
    az = 0.7 + float(mouse[0]) * 0.01
    el = 0.82 + float(mouse[1]) * 0.005
    dist = 6.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.2, 0.0)
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
    name="inductor",
    description="a toroidal inductor — copper wire wound in a helix around a dark "
                "ferrite ring, two leads dropping off, turning over a studio backdrop. "
                "It stores energy in a magnetic field and resists changes in current.",
    renderer=_render,
)
