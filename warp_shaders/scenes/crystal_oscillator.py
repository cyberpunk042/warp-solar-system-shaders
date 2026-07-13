"""Crystal oscillator — the quartz timekeeper in its metal can.

Squeeze a quartz crystal and it makes a voltage; apply a voltage and it flexes
(piezoelectricity). Cut a thin blank to size and it rings at one extremely
precise mechanical frequency, and a little feedback circuit keeps it ringing —
that steady tick is the clock every processor counts. The bright HC-49 can (left)
is the sealed package; beside it, the raw quartz blank it holds, shimmering as it
vibrates. Over a studio backdrop. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_cylinder, sd_round_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0
_CX = -0.95     # can x
_QX = 1.35      # quartz x


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _can(q: wp.vec3) -> float:
    return sd_round_box(q - wp.vec3(_CX, 0.05, 0.0), wp.vec3(0.44, 0.86, 0.26), 0.16)


@wp.func
def _quartz(q: wp.vec3) -> float:
    c = q - wp.vec3(_QX, 0.05, 0.0)
    # a slight lean so the blank catches the light
    ct = wp.cos(0.22)
    st = wp.sin(0.22)
    c = wp.vec3(ct * c[0] - st * c[2], c[1], st * c[0] + ct * c[2])
    return sd_round_box(c, wp.vec3(0.42, 0.66, 0.05), 0.05)


@wp.func
def _leads(q: wp.vec3) -> float:
    l0 = sd_cylinder(q - wp.vec3(_CX - 0.22, -1.5, 0.0), 0.75, 0.05)
    l1 = sd_cylinder(q - wp.vec3(_CX + 0.22, -1.5, 0.0), 0.75, 0.05)
    l2 = sd_cylinder(q - wp.vec3(_QX - 0.18, -1.35, 0.0), 0.6, 0.04)
    l3 = sd_cylinder(q - wp.vec3(_QX + 0.18, -1.35, 0.0), 0.6, 0.04)
    return wp.min(wp.min(l0, l1), wp.min(l2, l3))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_can(q), _quartz(q))
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
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dcan = _can(q)
    dq = _quartz(q)
    if dcan <= dq and dcan <= _leads(q):
        # aluminium can with a faint horizontal crimp seam
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        if wp.abs(q[1] - 0.55) < 0.02:
            col = col * 0.6
        img[i, j] = col
    elif dq <= _leads(q):
        # quartz blank: pale glass, shimmering emit (standing-wave vibration)
        base = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        base = wp.cw_mul(base, wp.vec3(1.4, 1.5, 1.7))
        shimmer = 0.5 + 0.5 * wp.sin(q[1] * 22.0 + time * 40.0)
        emit = wp.vec3(0.25, 0.45, 0.7) * (0.2 + 0.35 * shimmer)
        img[i, j] = base + emit
    else:
        img[i, j] = ec.lit(n, rd, 3, ao, wp.vec3(0.0, 0.0, 0.0))   # leads


def _render(width, height, time, mouse, device):
    az = 0.65 + float(mouse[0]) * 0.01
    el = 0.3 + float(mouse[1]) * 0.005
    dist = 7.0
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.35,
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
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="crystal_oscillator",
    description="a quartz crystal oscillator — the bright HC-49 metal can (left) "
                "sealed over the raw quartz blank (right) that vibrates at one precise "
                "frequency. Piezoelectric ringing is the clock every processor counts.",
    renderer=_render,
)
