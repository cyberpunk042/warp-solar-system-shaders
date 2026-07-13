"""IC package — a dual-in-line chip: the black body that holds a die.

The silicon die is fragile and microscopic, so it is glued into a lead-frame,
wired out to the legs, and sealed in moulded black epoxy — the familiar "chip".
This is a DIP (dual in-line package): two rows of tinned legs on 0.1-inch pitch,
a notch and a dimple marking pin 1 so you plug it in the right way round. Inside
is one die; the pins are the only way in or out. It turns over a studio backdrop.
See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder, sd_sphere
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0
_PITCH = 0.3
_HALFN = 3.5     # pins span x in [-1.05, 1.05]
_ZS = 0.52       # pin-row z offset


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.42
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _body(q: wp.vec3) -> float:
    b = sd_box(q, wp.vec3(1.32, 0.24, 0.5)) - 0.03
    # pin-1 notch carved from the -x end
    notch = sd_cylinder(wp.vec3(q[1], q[0] + 1.32, q[2]), 0.6, 0.16)
    return op_subtract(b, notch)


@wp.func
def _pins(q: wp.vec3) -> float:
    # repeat pins along x, clamp count to the body span
    xi = wp.floor(q[0] / _PITCH + 0.5)
    xi = wp.clamp(xi, -_HALFN, _HALFN)
    xr = q[0] - _PITCH * xi
    pz = sd_box(wp.vec3(xr, q[1] + 0.42, q[2] - _ZS), wp.vec3(0.045, 0.34, 0.05))
    nz = sd_box(wp.vec3(xr, q[1] + 0.42, q[2] + _ZS), wp.vec3(0.045, 0.34, 0.05))
    return wp.min(pz, nz)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_body(q), _pins(q))
    floor = p[1] + 1.1
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

    if p[1] < -1.07:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    if _body(q) <= _pins(q):
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))    # black epoxy
        # pin-1 dimple dot near the -x/-z corner on the top face
        if q[1] > 0.2:
            dd = wp.length(wp.vec2(q[0] + 0.95, q[2] + 0.28))
            if dd < 0.11:
                col = col * 0.35
        img[i, j] = col
    else:
        img[i, j] = ec.lit(n, rd, 3, ao, wp.vec3(0.0, 0.0, 0.0))   # tinned legs


def _render(width, height, time, mouse, device):
    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.5 + float(mouse[1]) * 0.005
    dist = 6.4
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.3,
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
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="ic_package",
    description="a DIP integrated circuit — a moulded black epoxy body with a "
                "pin-1 notch and dimple and two rows of tinned legs on 0.1-inch pitch, "
                "the sealed home of a silicon die. Turning over a studio backdrop.",
    renderer=_render,
)
