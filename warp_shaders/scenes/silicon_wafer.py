"""Silicon wafer — a polished disc patterned into a grid of dies.

A boule is sawn into paper-thin discs, lapped mirror-flat, and printed with
hundreds of identical chips (dies) in a rectangular grid; a flat notch on the rim
marks the crystal orientation. The stack of oxide and photoresist films on top is
only microns thick, so it splits reflected light into the shifting rainbow sheen
(thin-film interference) every wafer photo shows. Here the disc tilts under a
studio light, its die grid and iridescence catching the glare. See
``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import op_subtract, sd_box, sd_cylinder
from ..scene import Scene

_MAXD = 40.0
_RAD = 2.2
_TH = 0.06
_CELL = 0.34   # die pitch


@wp.func
def _wafer(p: wp.vec3) -> float:
    disc = sd_cylinder(p, _TH, _RAD)
    # orientation notch: carve a small box out of the +Z rim
    notch = sd_box(p - wp.vec3(0.0, 0.0, _RAD), wp.vec3(0.18, 0.3, 0.18))
    return op_subtract(disc, notch)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    a = time * 0.35
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    # tilt the wafer so we look across its face
    tb = 0.62
    ct = wp.cos(tb)
    st = wp.sin(tb)
    q = wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])
    wafer = _wafer(q)
    floor = p[1] + 2.0
    return wp.min(wafer, floor)


@wp.func
def _local(p: wp.vec3, time: float) -> wp.vec3:
    """Transform a world hit back into wafer-local coords (for the die grid)."""
    a = time * 0.35
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.62
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
        hr = 0.02 + 0.11 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.2 * occ, 0.0, 1.0)


@wp.func
def _irid(t: float) -> wp.vec3:
    # cosine palette — the thin-film rainbow
    return wp.vec3(0.5 + 0.5 * wp.cos(6.283 * (t + 0.0)),
                   0.5 + 0.5 * wp.cos(6.283 * (t + 0.33)),
                   0.5 + 0.5 * wp.cos(6.283 * (t + 0.66)))


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

    if p[1] < -1.97:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))  # glass floor
        return

    lp = _local(p, time)
    on_face = wp.abs(lp[1]) > (_TH - 0.012)   # top or bottom flat face
    base = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))

    if on_face and lp[1] > 0.0:
        # die grid: darken the scribe lines between cells
        gx = lp[0] / _CELL - wp.floor(lp[0] / _CELL)
        gz = lp[2] / _CELL - wp.floor(lp[2] / _CELL)
        line = wp.min(wp.min(gx, 1.0 - gx), wp.min(gz, 1.0 - gz))
        scribe = wp.clamp(line / 0.06, 0.0, 1.0)        # 0 on the lines
        base = base * (0.55 + 0.45 * scribe)
        # thin-film iridescence keyed on view angle + radius
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 2.5)
        rr = wp.sqrt(lp[0] * lp[0] + lp[2] * lp[2])
        tint = _irid(fres * 1.6 + rr * 0.35 + 0.1)
        base = base + tint * (fres * 0.32)

    img[i, j] = base


def _render(width, height, time, mouse, device):
    az = 0.85 + float(mouse[0]) * 0.01
    el = 0.55 + float(mouse[1]) * 0.005
    dist = 6.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.5,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(40.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.9, strength=0.28)


SCENE = Scene(
    name="silicon_wafer",
    description="a polished silicon wafer — a mirror disc printed with a grid of "
                "dies and a rim orientation notch, its thin oxide films splitting the "
                "studio glare into a shifting rainbow sheen. Slowly turning.",
    renderer=_render,
)
