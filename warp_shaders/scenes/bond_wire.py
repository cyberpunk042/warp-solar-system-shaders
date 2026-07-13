"""Bond wires — the die on its leadframe, wired out with gold.

Crack open a chip and this is inside: the silicon die glued to a metal leadframe
pad, and from tiny aluminium pads along the die edge, hair-thin gold wires arc out
to the leadframe fingers that become the package pins. A machine welds each wire
in a fraction of a second — thousands per chip on a big part. This is the actual
electrical bridge from the microscopic circuit to the outside world. Lit like a
die-shot under the microscope. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box, sd_capsule
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.55
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _pad(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.08, 0.0), wp.vec3(0.62, 0.05, 0.72)) - 0.02


@wp.func
def _die(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, 0.04, 0.0), wp.vec3(0.48, 0.05, 0.58)) - 0.01


@wp.func
def _fingers(q: wp.vec3) -> float:
    d = float(1e9)
    for k in range(4):
        zk = -0.78 + float(k) * 0.52
        fl = sd_box(q - wp.vec3(-1.25, -0.1, zk), wp.vec3(0.4, 0.045, 0.13)) - 0.01
        fr = sd_box(q - wp.vec3(1.25, -0.1, zk), wp.vec3(0.4, 0.045, 0.13)) - 0.01
        d = wp.min(d, wp.min(fl, fr))
    return d


@wp.func
def _wires(q: wp.vec3) -> float:
    d = float(1e9)
    rw = 0.022
    for k in range(4):
        zk = -0.78 + float(k) * 0.52
        # left wire: die edge -> apex -> finger
        a0 = wp.vec3(-0.46, 0.12, zk)
        c0 = wp.vec3(-1.05, -0.05, zk)
        b0 = (a0 + c0) * 0.5 + wp.vec3(0.0, 0.42, 0.0)
        d = wp.min(d, wp.min(sd_capsule(q, a0, b0, rw), sd_capsule(q, b0, c0, rw)))
        # right wire
        a1 = wp.vec3(0.46, 0.12, zk)
        c1 = wp.vec3(1.05, -0.05, zk)
        b1 = (a1 + c1) * 0.5 + wp.vec3(0.0, 0.42, 0.0)
        d = wp.min(d, wp.min(sd_capsule(q, a1, b1, rw), sd_capsule(q, b1, c1, rw)))
    return d


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pad(q), _die(q))
    d = op_union(d, _fingers(q))
    d = op_union(d, _wires(q))
    floor = p[1] + 0.9
    return wp.min(d, floor)


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0012
    dx = _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time)
    dz = _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.015 + 0.07 * float(k)
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
        if d < 0.0006 * t + 0.0003:
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

    if p[1] < -0.87:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dd = _die(q)
    dw = _wires(q)
    dp = _pad(q)
    df = _fingers(q)
    mind = wp.min(wp.min(dd, dw), wp.min(dp, df))
    if dw <= mind + 0.001:
        img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.02, 0.015, 0.0))   # gold wire
    elif dd <= mind + 0.001:
        # silicon die with a faint bond-pad + circuitry sheen
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        if q[1] > 0.06:
            gx = q[0] / 0.09 - wp.floor(q[0] / 0.09)
            gz = q[2] / 0.09 - wp.floor(q[2] / 0.09)
            grid = wp.min(wp.min(gx, 1.0 - gx), wp.min(gz, 1.0 - gz))
            if grid < 0.12:
                col = col * 1.6 + wp.vec3(0.04, 0.05, 0.08)
        img[i, j] = col
    else:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))      # copper leadframe


def _render(width, height, time, mouse, device):
    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.5 + float(mouse[1]) * 0.005
    dist = 5.0
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.25,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(40.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.4)


SCENE = Scene(
    name="bond_wire",
    description="the inside of a chip — a silicon die on a copper leadframe pad, "
                "hair-thin gold wires arcing from the die's bond pads out to the "
                "leadframe fingers. The electrical bridge from microscopic circuit to pin.",
    renderer=_render,
)
