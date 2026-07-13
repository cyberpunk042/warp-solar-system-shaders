"""LED + diode — a pn junction that emits light, and one that blocks reverse flow.

A diode is a single p-n junction: current passes one way and is blocked the other
(note the cathode band on the little glass signal diode). Make that junction out
of a direct-bandgap crystal and the energy released when electrons and holes
recombine comes out as photons instead of heat — a light-emitting diode. Here the
5 mm epoxy dome glows from the die on its leadframe cup, the longer lead marking
the anode. Both float over a studio backdrop. See
``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import (
    op_smooth_union, op_union, sd_cylinder, sd_sphere,
)
from ..scene import Scene

_MAXD = 40.0
_LX = -0.95     # LED x
_LR = 0.5       # LED body radius
_DX = 1.5       # diode x


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _led_body(q: wp.vec3) -> float:
    c = q - wp.vec3(_LX, 0.0, 0.0)
    barrel = sd_cylinder(c - wp.vec3(0.0, -0.1, 0.0), 0.55, _LR)
    dome = sd_sphere(c - wp.vec3(0.0, 0.45, 0.0), _LR)
    body = op_smooth_union(barrel, dome, 0.18)
    flange = sd_cylinder(c - wp.vec3(0.0, -0.62, 0.0), 0.08, _LR + 0.12)
    return op_union(body, flange)


@wp.func
def _led_die(q: wp.vec3) -> float:
    return sd_sphere(q - wp.vec3(_LX, -0.15, 0.0), 0.1)


@wp.func
def _diode(q: wp.vec3) -> float:
    c = q - wp.vec3(_DX, 0.0, 0.0)
    return sd_cylinder(wp.vec3(c[1], c[0], c[2]), 0.55, 0.24) - 0.03


@wp.func
def _leads(q: wp.vec3) -> float:
    a0 = sd_cylinder(q - wp.vec3(_LX - 0.18, -1.55, 0.0), 0.95, 0.05)   # anode (long)
    a1 = sd_cylinder(q - wp.vec3(_LX + 0.18, -1.35, 0.0), 0.75, 0.05)
    d0 = sd_cylinder(wp.vec3(q[1], q[0] - (_DX - 1.5), q[2]) - wp.vec3(0.0, _DX + 0.7, 0.0),
                     0.7, 0.045)
    d1 = sd_cylinder(wp.vec3(q[1], q[0] - (_DX - 1.5), q[2]) - wp.vec3(0.0, -(_DX + 0.7), 0.0),
                     0.7, 0.045)
    return wp.min(wp.min(a0, a1), wp.min(d0, d1))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_led_body(q), _diode(q))
    d = op_union(d, _leads(q))
    floor = p[1] + 1.5
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

    if p[1] < -1.47:
        # floor picks up a red pool of LED light near the LED base
        base = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        wq = _rot(p, time)
        dd = wp.length(wq - wp.vec3(_LX, -1.5, 0.0))
        base = base + wp.vec3(0.9, 0.12, 0.08) * (0.5 * wp.exp(-dd * 1.4))
        img[i, j] = base
        return

    q = _rot(p, time)
    db = _led_body(q)
    dd = _diode(q)

    if db < 0.03 and db <= dd:
        # translucent epoxy: tint red, glow brightest near the die on the axis
        c = q - wp.vec3(_LX, 0.0, 0.0)
        axis_r = wp.sqrt(c[0] * c[0] + c[2] * c[2])
        core = wp.exp(-axis_r * 2.2) * wp.clamp(1.0 - c[1] * 0.4, 0.2, 1.0)
        base = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        red = wp.vec3(1.0, 0.16, 0.10)
        emit = red * (0.55 + 1.3 * core)
        img[i, j] = wp.cw_mul(base, red * 1.5) + emit
    elif dd < 0.03:
        # black glass diode with a silver cathode band near +x end
        col = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        col = col * 0.5
        if q[0] - _DX > 0.32:
            col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))   # cathode band
        img[i, j] = col
    else:
        img[i, j] = ec.lit(n, rd, 3, ao, wp.vec3(0.0, 0.0, 0.0))   # leads


def _render(width, height, time, mouse, device):
    az = 0.65 + float(mouse[0]) * 0.01
    el = 0.26 + float(mouse[1]) * 0.005
    dist = 7.0
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.3,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.1, -0.05, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.4)


SCENE = Scene(
    name="led",
    description="a glowing 5 mm red LED (epoxy dome lit from the die on its "
                "leadframe, longer anode lead) beside a black glass signal diode with "
                "a silver cathode band — a pn junction emitting light, and one blocking "
                "reverse current. Floating over a studio backdrop.",
    renderer=_render,
)
