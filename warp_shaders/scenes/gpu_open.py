"""GPU (fanless / open) — a passive graphics card with the guts exposed.

The same board with the shroud and fans stripped away: a low passive heatsink over
the GPU (no fan — it sheds heat by convection alone, so fanless cards run slower and
cooler), and everything else laid bare. You can see the GPU die, the ring of GDDR
memory chips, the VRM — chokes, MOSFETs, and bulk capacitors that turn the 12 V rail
into the low, heavy current the die drinks — the gold PCIe edge fingers, the power
connector, and the copper traces wiring them all together. This is the board to watch
current flow across. See ``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 55.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.12 + 0.06 * wp.sin(time * 0.35)
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.12
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    board = sd_box(q, wp.vec3(3.3, 0.06, 1.25)) - 0.01
    notch = sd_box(q - wp.vec3(1.0, 0.0, -1.25), wp.vec3(0.1, 0.2, 0.16))
    return op_subtract(board, notch)


@wp.func
def _gpu(q: wp.vec3) -> float:
    sub = sd_box(q - wp.vec3(-0.5, 0.12, 0.1), wp.vec3(0.85, 0.06, 0.8)) - 0.01
    die = sd_box(q - wp.vec3(-0.5, 0.19, 0.1), wp.vec3(0.6, 0.05, 0.55)) - 0.005
    return op_union(sub, die)


@wp.func
def _die_only(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(-0.5, 0.19, 0.1), wp.vec3(0.6, 0.05, 0.55)) - 0.005


@wp.func
def _sink(q: wp.vec3) -> float:
    # low passive fin stack sitting on the GPU die
    c = q - wp.vec3(-0.5, 0.34, 0.1)
    xi = wp.clamp(wp.floor(c[0] / 0.13 + 0.5), -4.0, 4.0)
    xr = c[0] - 0.13 * xi
    return sd_box(wp.vec3(xr, c[1], c[2]), wp.vec3(0.03, 0.12, 0.56))


@wp.func
def _mem(q: wp.vec3) -> float:
    # six GDDR packages ringing the GPU
    m0 = sd_box(q - wp.vec3(-0.5, 0.11, 1.02), wp.vec3(0.28, 0.05, 0.2)) - 0.008
    m1 = sd_box(q - wp.vec3(-0.5, 0.11, -0.82), wp.vec3(0.28, 0.05, 0.2)) - 0.008
    m2 = sd_box(q - wp.vec3(-1.55, 0.11, 0.45), wp.vec3(0.2, 0.05, 0.28)) - 0.008
    m3 = sd_box(q - wp.vec3(-1.55, 0.11, -0.25), wp.vec3(0.2, 0.05, 0.28)) - 0.008
    m4 = sd_box(q - wp.vec3(0.55, 0.11, 0.45), wp.vec3(0.2, 0.05, 0.28)) - 0.008
    m5 = sd_box(q - wp.vec3(0.55, 0.11, -0.25), wp.vec3(0.2, 0.05, 0.28)) - 0.008
    return wp.min(wp.min(wp.min(m0, m1), wp.min(m2, m3)), wp.min(m4, m5))


@wp.func
def _choke(q: wp.vec3) -> float:
    xi = wp.clamp(wp.floor((q[0] - 1.4) / 0.42 + 0.5), 0.0, 3.0)
    x = 1.4 + 0.42 * xi
    return sd_box(q - wp.vec3(x, 0.15, 0.75), wp.vec3(0.16, 0.1, 0.16)) - 0.01


@wp.func
def _caps(q: wp.vec3) -> float:
    xi = wp.clamp(wp.floor((q[0] - 1.5) / 0.3 + 0.5), 0.0, 3.0)
    x = 1.5 + 0.3 * xi
    return sd_cylinder(q - wp.vec3(x, 0.16, 0.05), 0.12, 0.11)


@wp.func
def _power(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(2.7, 0.16, -0.75), wp.vec3(0.5, 0.12, 0.22)) - 0.01


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pcb(q), _gpu(q))
    d = op_union(d, _sink(q))
    d = op_union(d, _mem(q))
    d = op_union(d, _choke(q))
    d = op_union(d, _caps(q))
    return op_union(d, _power(q))


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
        hr = 0.015 + 0.08 * float(k)
        d = _map(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.func
def _trace(lx: float, lz: float) -> float:
    c = float(0.0)
    w = 0.03
    # GPU <-> memory + VRM buses
    if wp.abs(lz - 0.55) < w and lx > -1.4 and lx < 0.5:
        c = 1.0
    if wp.abs(lz + 0.5) < w and lx > -1.4 and lx < 0.5:
        c = 1.0
    if wp.abs(lx - 0.5) < w and lz > -0.6 and lz < 0.9:
        c = 1.0
    if wp.abs(lz - 0.9) < w and lx > 0.4 and lx < 2.6:
        c = 1.0
    if wp.abs(lz - 0.05) < w and lx > 0.5 and lx < 2.6:
        c = 1.0
    return c


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
    for _ in range(220):
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
    dpcb = _pcb(q)
    dgpu = _gpu(q)
    dsink = _sink(q)
    dmem = _mem(q)
    dch = _choke(q)
    dca = _caps(q)
    dpw = _power(q)
    mind = wp.min(wp.min(wp.min(dpcb, dgpu), wp.min(dsink, dmem)),
                  wp.min(wp.min(dch, dca), dpw))
    eps = 0.001
    if dsink <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))       # aluminium fins
    elif dgpu <= mind + eps:
        if _die_only(q) <= dgpu + eps and q[1] > 0.19:
            col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
            gx = q[0] / 0.11 - wp.floor(q[0] / 0.11)
            gz = q[2] / 0.11 - wp.floor(q[2] / 0.11)
            grid = wp.min(wp.min(gx, 1.0 - gx), wp.min(gz, 1.0 - gz))
            if grid < 0.13:
                col = col * 1.5 + wp.vec3(0.05, 0.07, 0.12)
            img[i, j] = col                                            # GPU die
        else:
            img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.7  # substrate
    elif dmem <= mind + eps:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))       # GDDR chips
    elif dch <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.8  # VRM chokes
    elif dca <= mind + eps:
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(0.7, 0.8, 1.0))            # cap cans
    elif dpw <= mind + eps:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))       # power connector
    else:
        # PCB: green, gold traces on top, gold PCIe fingers on the -z edge
        if q[2] < -1.05 and q[1] > -0.02:
            fx = q[0] / 0.12 - wp.floor(q[0] / 0.12)
            if fx > 0.3 and wp.abs(q[0] - 1.0) > 0.16:
                img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))
            else:
                img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))
        elif q[1] > 0.05 and n[1] > 0.5 and _trace(q[0], q[2]) > 0.5:
            img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))   # gold trace
        else:
            img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))   # green mask


def _render(width, height, time, mouse, device):
    az = 0.12 + float(mouse[0]) * 0.01
    el = 0.62 + float(mouse[1]) * 0.005
    dist = 7.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, -0.05, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.7, strength=0.3)


SCENE = Scene(
    name="gpu_open",
    description="a fanless graphics card with the guts exposed — a low passive "
                "heatsink on the GPU die, a ring of GDDR memory chips, the VRM (chokes, "
                "caps), gold PCIe fingers, a power connector, and copper traces wiring "
                "it all together. The board to watch current flow across.",
    renderer=_render,
)
