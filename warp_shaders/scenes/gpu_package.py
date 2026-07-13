"""GPU package — a big graphics processor die with its memory around it.

A GPU is a huge silicon die — thousands of the simple ALU/logic cells packed into
a grid of shader cores (see `gpu_floorplan`) — and it needs enormous memory
bandwidth to feed them, so the memory is mounted right next to it on the same
package: stacks of DRAM (GDDR, or HBM towers sitting on a silicon interposer). This
is the bare package before the cooler goes on: the exposed central die flanked by
four memory stacks on a dark substrate, ringed by supply capacitors. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 45.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.4 + 0.12 * wp.sin(time * 0.5)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _sub(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(2.15, 0.1, 1.6)) - 0.02


@wp.func
def _die(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, 0.16, 0.0), wp.vec3(0.95, 0.06, 0.82)) - 0.005


@wp.func
def _hbm(q: wp.vec3) -> float:
    a = sd_box(q - wp.vec3(-1.5, 0.19, 0.62), wp.vec3(0.33, 0.11, 0.42)) - 0.01
    b = sd_box(q - wp.vec3(-1.5, 0.19, -0.62), wp.vec3(0.33, 0.11, 0.42)) - 0.01
    c = sd_box(q - wp.vec3(1.5, 0.19, 0.62), wp.vec3(0.33, 0.11, 0.42)) - 0.01
    d = sd_box(q - wp.vec3(1.5, 0.19, -0.62), wp.vec3(0.33, 0.11, 0.42)) - 0.01
    return wp.min(wp.min(a, b), wp.min(c, d))


@wp.func
def _caps(q: wp.vec3) -> float:
    xi = wp.clamp(wp.floor(q[0] / 0.34 + 0.5), -5.0, 5.0)
    cx = 0.34 * xi
    f = sd_box(q - wp.vec3(cx, 0.12, 1.4), wp.vec3(0.06, 0.03, 0.09))
    b = sd_box(q - wp.vec3(cx, 0.12, -1.4), wp.vec3(0.06, 0.03, 0.09))
    zi = wp.clamp(wp.floor(q[2] / 0.34 + 0.5), -3.0, 3.0)
    cz = 0.34 * zi
    lft = sd_box(q - wp.vec3(-2.0, 0.12, cz), wp.vec3(0.09, 0.03, 0.06))
    rgt = sd_box(q - wp.vec3(2.0, 0.12, cz), wp.vec3(0.09, 0.03, 0.06))
    return wp.min(wp.min(f, b), wp.min(lft, rgt))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_sub(q), _die(q))
    d = op_union(d, _hbm(q))
    return op_union(d, _caps(q))


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

    q = _rot(p, time)
    ddie = _die(q)
    dhbm = _hbm(q)
    dcaps = _caps(q)
    dsub = _sub(q)
    mind = wp.min(wp.min(ddie, dhbm), wp.min(dcaps, dsub))
    eps = 0.001
    if ddie <= mind + eps:
        # big GPU die: silicon with a fine shader-core grid sheen
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.16:
            gx = q[0] / 0.13 - wp.floor(q[0] / 0.13)
            gz = q[2] / 0.13 - wp.floor(q[2] / 0.13)
            grid = wp.min(wp.min(gx, 1.0 - gx), wp.min(gz, 1.0 - gz))
            if grid < 0.12:
                col = col * 1.5 + wp.vec3(0.05, 0.07, 0.12)
        img[i, j] = col
    elif dhbm <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        col = wp.cw_mul(col, wp.vec3(1.15, 1.15, 1.25))            # memory stacks
        if q[1] > 0.28:
            mx = q[0] / 0.11 - wp.floor(q[0] / 0.11)               # stacked-die layer lines
            if mx < 0.14:
                col = col * 0.7
        img[i, j] = col
    elif dcaps <= mind + eps:
        col = ec.lit(n, rd, 6, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(1.1, 0.85, 0.6))         # tan caps
    else:
        # dark green substrate with gold traces fanning die -> memory
        col = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.7
        if q[1] > 0.06 and n[1] > 0.5:
            tz = q[2] / 0.09 - wp.floor(q[2] / 0.09)
            tx = q[0] / 0.09 - wp.floor(q[0] / 0.09)
            on = float(0.0)
            if wp.abs(q[0]) > 1.0 and tz < 0.32:
                on = 1.0                                            # horizontal buses to L/R memory
            if wp.abs(q[2]) > 0.85 and tx < 0.32:
                on = 1.0                                            # vertical buses to F/B
            if on > 0.5:
                col = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))  # gold trace
        img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.4 + float(mouse[0]) * 0.01
    el = 0.58 + float(mouse[1]) * 0.005
    dist = 7.0
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
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
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="gpu_package",
    description="a bare GPU package — a big central graphics die (fine shader-core "
                "sheen) flanked by four memory stacks on a dark substrate, ringed by "
                "supply capacitors. The processor before the cooler goes on.",
    renderer=_render,
)
