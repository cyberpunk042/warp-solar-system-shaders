"""BGA — a ball grid array package, tilted to show its solder-ball underside.

A DIP's legs run out of pins fast. A ball grid array instead studs the whole
underside of the package with a grid of solder balls — hundreds of connections in
the same area, and short fat ones that carry fast signals and heat better. At
reflow the balls melt and self-align the chip to matching pads. This one is tipped
up so you can see the array; the balls are shiny solder, the body moulded epoxy.
Over a studio backdrop. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box, sd_sphere
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0
_HW = 1.05       # package half-width
_HT = 0.17       # package half-thickness
_PITCH = 0.28
_HALFN = 3.0     # balls span +/- _HALFN*_PITCH
_BR = 0.11       # ball radius


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    # tip the part over so the ball underside faces up toward the elevated camera
    tb = -2.75
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _body(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(_HW, _HT, _HW)) - 0.02


@wp.func
def _balls(q: wp.vec3) -> float:
    xi = wp.clamp(wp.floor(q[0] / _PITCH + 0.5), -_HALFN, _HALFN)
    zi = wp.clamp(wp.floor(q[2] / _PITCH + 0.5), -_HALFN, _HALFN)
    bx = q[0] - _PITCH * xi
    bz = q[2] - _PITCH * zi
    return sd_sphere(wp.vec3(bx, q[1] + _HT + _BR * 0.6, bz), _BR)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_body(q), _balls(q))
    floor = p[1] + 1.7
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
        hr = 0.02 + 0.08 * float(k)
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

    if p[1] < -1.67:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    if _balls(q) <= _body(q):
        img[i, j] = ec.lit(n, rd, 3, ao, wp.vec3(0.0, 0.0, 0.0))   # solder ball
    else:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))         # epoxy body
        # pin-1 dot on the +y (top) face, one corner
        if q[1] > _HT - 0.02:
            dd = wp.length(wp.vec2(q[0] + 0.82, q[2] + 0.82))
            if dd < 0.13:
                col = col * 0.4
        img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.7 + float(mouse[0]) * 0.01
    el = 0.72 + float(mouse[1]) * 0.005
    dist = 6.8
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.3,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.15, 0.0)
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
    name="bga",
    description="a ball grid array package, tipped up to show its underside — a "
                "full grid of shiny solder balls instead of legs, hundreds of short "
                "connections that self-align at reflow. Moulded epoxy body, over a studio backdrop.",
    renderer=_render,
)
