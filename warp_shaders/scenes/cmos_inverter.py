"""CMOS inverter — a PMOS and an NMOS transistor: the atom of digital logic.

Every logic gate in a processor is built from complementary pairs. Here a PMOS
transistor (warm/p-type, tied to the VDD rail) sits above an NMOS (cool/n-type,
tied to ground); their gates are wired together as the input, their drains
together as the output. Put the input high and the NMOS pulls the output to 0;
put it low and the PMOS pulls it to 1 — a NOT gate. Because only one transistor
conducts at rest, it burns almost no power idle: that is why CMOS won. The output
rail glows here. See ``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 40.0


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.5
    ct = wp.cos(tb)
    st = wp.sin(tb)
    return wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])


@wp.func
def _sub(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.5, 0.0), wp.vec3(1.9, 0.16, 1.25)) - 0.02


@wp.func
def _pmos_pads(q: wp.vec3) -> float:
    # PMOS = one warm raised block on the left
    return sd_box(q - wp.vec3(-0.85, -0.24, 0.0), wp.vec3(0.42, 0.14, 0.7)) - 0.02


@wp.func
def _nmos_pads(q: wp.vec3) -> float:
    # NMOS = one cool raised block on the right
    return sd_box(q - wp.vec3(0.85, -0.24, 0.0), wp.vec3(0.42, 0.14, 0.7)) - 0.02


@wp.func
def _gates(q: wp.vec3) -> float:
    # a poly gate bar over each block, joined by the input line (all one node)
    gp = sd_box(q - wp.vec3(-0.85, 0.0, 0.0), wp.vec3(0.09, 0.12, 0.78))
    gn = sd_box(q - wp.vec3(0.85, 0.0, 0.0), wp.vec3(0.09, 0.12, 0.78))
    link = sd_box(q - wp.vec3(0.0, 0.06, -0.78), wp.vec3(0.95, 0.05, 0.06))
    return wp.min(wp.min(gp, gn), link)


@wp.func
def _rails(q: wp.vec3) -> float:
    vdd = sd_box(q - wp.vec3(-1.55, 0.0, 0.0), wp.vec3(0.06, 0.05, 0.7))    # VDD to PMOS
    gnd = sd_box(q - wp.vec3(1.55, 0.0, 0.0), wp.vec3(0.06, 0.05, 0.7))     # GND to NMOS
    inp = sd_box(q - wp.vec3(0.0, 0.06, -1.05), wp.vec3(0.05, 0.05, 0.32))  # input stub
    return wp.min(wp.min(vdd, gnd), inp)


@wp.func
def _outnode(q: wp.vec3) -> float:
    # output metal joining the two inner drains, glows
    return sd_box(q - wp.vec3(0.0, 0.0, 0.62), wp.vec3(0.95, 0.05, 0.07))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_sub(q), _pmos_pads(q))
    d = op_union(d, _nmos_pads(q))
    d = op_union(d, _gates(q))
    d = op_union(d, _rails(q))
    d = op_union(d, _outnode(q))
    floor = p[1] + 0.75
    return wp.min(d, floor)


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
        hr = 0.015 + 0.08 * float(k)
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
        if d < 0.0007 * t + 0.0003:
            hit = 1
            break
        t += d * 0.88
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, time)
    ao = _ao(p, n, time)

    if p[1] < -0.72:
        img[i, j] = ec.lit(n, rd, 8, ao, wp.vec3(0.0, 0.0, 0.0))
        return

    q = _rot(p, time)
    dp = _pmos_pads(q)
    dn = _nmos_pads(q)
    dg = _gates(q)
    dr = _rails(q)
    do = _outnode(q)
    ds = _sub(q)
    mind = wp.min(wp.min(wp.min(dp, dn), wp.min(dg, dr)), wp.min(do, ds))
    eps = 0.001
    if do <= mind + eps:
        col = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = col + wp.vec3(0.3, 0.9, 0.5) * 0.5              # glowing output node
    elif dr <= mind + eps:
        img[i, j] = ec.lit(n, rd, 1, ao, wp.vec3(0.0, 0.0, 0.0))    # copper rails
    elif dg <= mind + eps:
        col = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(2.4, 1.2, 1.1))          # poly gate (input)
    elif dp <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(2.4, 1.15, 0.45))       # PMOS p+ (amber)
    elif dn <= mind + eps:
        col = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))
        img[i, j] = wp.cw_mul(col, wp.vec3(0.8, 1.1, 1.7))        # NMOS n+ (cool)
    else:
        img[i, j] = ec.lit(n, rd, 0, ao, wp.vec3(0.0, 0.0, 0.0))  # substrate


def _render(width, height, time, mouse, device):
    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.62 + float(mouse[1]) * 0.005
    dist = 5.8
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
    return ec.finish(img.numpy(), width, height, threshold=1.4)


SCENE = Scene(
    name="cmos_inverter",
    description="a CMOS inverter — a warm PMOS (to VDD) and cool NMOS (to ground) "
                "sharing a poly gate as input and a glowing metal node as output. The "
                "NOT gate, and the atom every logic circuit in a processor is built from.",
    renderer=_render,
)
