"""Heatsink — a finned cooler with copper heat pipes.

A processor turns its whole power budget into heat in a fingernail-sized die, so it
needs somewhere to dump it. A heatsink is a big folded stack of thin aluminium fins
— enormous surface area for air to carry heat away — fed by copper **heat pipes**:
sealed tubes where fluid boils at the hot base, rises, condenses in the cool fins,
and wicks back, moving heat far faster than solid metal. This tower sits on the CPU's
heat spreader; a fan (not shown) blows through the fins. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box, sd_cylinder, sd_sphere
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 50.0
_FIN_PITCH = 0.15
_FIN_N = 8.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.5 + 0.14 * wp.sin(time * 0.5)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _base(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.85, 0.0), wp.vec3(1.35, 0.14, 0.95)) - 0.02


@wp.func
def _fins(q: wp.vec3) -> float:
    xi = wp.clamp(wp.floor(q[0] / _FIN_PITCH + 0.5), -_FIN_N, _FIN_N)
    xr = q[0] - _FIN_PITCH * xi
    return sd_box(wp.vec3(xr, q[1] - 0.15, q[2]), wp.vec3(0.028, 0.68, 0.9))


@wp.func
def _pipes(q: wp.vec3) -> float:
    p0 = sd_cylinder(q - wp.vec3(-0.55, 0.05, 0.35), 0.95, 0.1)
    t0 = sd_sphere(q - wp.vec3(-0.55, 1.0, 0.35), 0.1)
    p1 = sd_cylinder(q - wp.vec3(0.0, 0.05, -0.3), 0.95, 0.1)
    t1 = sd_sphere(q - wp.vec3(0.0, 1.0, -0.3), 0.1)
    p2 = sd_cylinder(q - wp.vec3(0.55, 0.05, 0.35), 0.95, 0.1)
    t2 = sd_sphere(q - wp.vec3(0.55, 1.0, 0.35), 0.1)
    return wp.min(wp.min(op_union(p0, t0), op_union(p1, t1)), op_union(p2, t2))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_base(q), _fins(q))
    return op_union(d, _pipes(q))


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
        hr = 0.02 + 0.08 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.2 * occ, 0.0, 1.0)


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
    for _ in range(210):
        p = eye + rd * t
        d = _map(p, time)
        if d < 0.0007 * t + 0.0004:
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

    q = _rot(p, time)
    dp = _pipes(q)
    df = _fins(q)
    db = _base(q)
    if dp <= df and dp <= db:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))     # copper heat pipe
    else:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))     # aluminium fins/base


def _render(width, height, time, mouse, device):
    az = 0.55 + float(mouse[0]) * 0.01
    el = 0.32 + float(mouse[1]) * 0.005
    dist = 6.8
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.35,
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
    return ec.finish(img.numpy(), width, height, threshold=1.6)


SCENE = Scene(
    name="heatsink",
    description="a CPU/GPU cooler — a stack of thin aluminium fins fed by copper "
                "heat pipes rising from the base, huge surface area to shed a "
                "processor's heat into the air. Floated on a studio backdrop.",
    renderer=_render,
)
