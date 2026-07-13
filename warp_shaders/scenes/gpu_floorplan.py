"""GPU floorplan — inside the die: a grid of shader cores lighting up.

Zoom into the graphics die and this is the layout: a huge regular **grid of shader
cores** (streaming multiprocessors), each a little bundle of the ALU/logic cells
from the components round, all running the same instruction on different data —
that is what makes a GPU fast at graphics and at the massively parallel maths that
runs on it. A shared **cache** spine runs down the middle and **memory controllers**
line the edges, talking to the GDDR. The cores flicker here as work flows across
them — the compute fabric a "virtual graphics card" would model. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 45.0
_PITCH = 0.42
_NX = 6.0        # columns span +/- _NX
_NZ = 4.0        # rows span +/- _NZ


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.2 + 0.1 * wp.sin(time * 0.3)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _die(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(_NX * _PITCH + 0.28, 0.09, _NZ * _PITCH + 0.28)) - 0.02


@wp.func
def _blocks(q: wp.vec3) -> float:
    ix = wp.clamp(wp.floor(q[0] / _PITCH + 0.5), -_NX, _NX)
    iz = wp.clamp(wp.floor(q[2] / _PITCH + 0.5), -_NZ, _NZ)
    cx = q[0] - _PITCH * ix
    cz = q[2] - _PITCH * iz
    return sd_box(wp.vec3(cx, q[1] - 0.12, cz), wp.vec3(0.17, 0.05, 0.17))


@wp.func
def _hash(ix: float, iz: float) -> float:
    h = wp.sin(ix * 12.9898 + iz * 78.233) * 43758.5453
    return h - wp.floor(h)


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    return wp.min(_die(q), _blocks(q))


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
    for k in range(4):
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

    q = _rot(p, time)
    if _blocks(q) <= _die(q):
        ix = wp.clamp(wp.floor(q[0] / _PITCH + 0.5), -_NX, _NX)
        iz = wp.clamp(wp.floor(q[2] / _PITCH + 0.5), -_NZ, _NZ)
        base = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.5
        top = float(0.0)
        if q[1] > 0.13 and n[1] > 0.5:
            top = 1.0
        if wp.abs(ix) > _NX - 0.5:
            # memory controllers on the left/right edges — steady blue
            img[i, j] = base + wp.vec3(0.1, 0.35, 0.9) * (0.4 * top + 0.1)
        elif wp.abs(ix) < 0.5:
            # cache spine down the middle — amber
            img[i, j] = base + wp.vec3(1.0, 0.65, 0.15) * (0.5 * top + 0.12)
        else:
            # shader cores — cyan, flickering with activity
            ph = _hash(ix, iz)
            act = 0.5 + 0.5 * wp.sin(time * 2.2 + ph * 6.283)
            act = act * act
            img[i, j] = base + wp.vec3(0.15, 0.85, 0.95) * (top * (0.15 + 0.85 * act) + 0.05)
    else:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.6   # dark die floor


def _render(width, height, time, mouse, device):
    az = 0.3 + float(mouse[0]) * 0.01
    el = 0.92 + float(mouse[1]) * 0.005
    dist = 6.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.1,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.5, strength=0.4)


SCENE = Scene(
    name="gpu_floorplan",
    description="inside a GPU die — a grid of shader cores (cyan, flickering with "
                "activity) around a central cache spine (amber) with memory controllers "
                "on the edges (blue). The massively-parallel compute fabric, seen from above.",
    renderer=_render,
)
