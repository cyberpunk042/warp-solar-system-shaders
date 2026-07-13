"""Silicon crystal — the diamond-cubic lattice.

Silicon crystallises in the diamond-cubic structure: every atom sits at the
centre of a tetrahedron and shares a covalent bond with each of its four nearest
neighbours (the sp3 bonds), and those neighbours are themselves centres of the
same motif — two interpenetrating face-centred-cubic lattices. This shows the
fundamental cell: a central atom, its four bonded neighbours, and the bonds
carrying on outward to where the crystal continues. See
``docs/research/35-electronics-components.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..procedural.sdf import op_union, sd_capsule, sd_sphere
from ..scene import Scene

_MAXD = 30.0
_R_ATOM = 0.34
_R_BOND = 0.10
_B = 1.15  # bond half-length (neighbour distance)

# the four sp3 tetrahedral directions (unit)
_T0 = wp.constant(wp.vec3(0.57735, 0.57735, 0.57735))
_T1 = wp.constant(wp.vec3(0.57735, -0.57735, -0.57735))
_T2 = wp.constant(wp.vec3(-0.57735, 0.57735, -0.57735))
_T3 = wp.constant(wp.vec3(-0.57735, -0.57735, 0.57735))


@wp.func
def _atom(p: wp.vec3, c: wp.vec3, r: float) -> float:
    return sd_sphere(p - c, r)


@wp.func
def _lattice(p: wp.vec3) -> float:
    d = _atom(p, wp.vec3(0.0, 0.0, 0.0), _R_ATOM)
    n0 = _T0 * _B
    n1 = _T1 * _B
    n2 = _T2 * _B
    n3 = _T3 * _B
    # neighbour atoms
    d = op_union(d, _atom(p, n0, _R_ATOM))
    d = op_union(d, _atom(p, n1, _R_ATOM))
    d = op_union(d, _atom(p, n2, _R_ATOM))
    d = op_union(d, _atom(p, n3, _R_ATOM))
    # central bonds
    o = wp.vec3(0.0, 0.0, 0.0)
    d = op_union(d, sd_capsule(p, o, n0, _R_BOND))
    d = op_union(d, sd_capsule(p, o, n1, _R_BOND))
    d = op_union(d, sd_capsule(p, o, n2, _R_BOND))
    d = op_union(d, sd_capsule(p, o, n3, _R_BOND))
    # each neighbour continues the lattice: three stub bonds pointing away,
    # rotated tetrahedron (the inverted set) so the crystal reads as ongoing
    d = op_union(d, sd_capsule(p, n0, n0 + _T1 * _B, _R_BOND))
    d = op_union(d, sd_capsule(p, n0, n0 + _T2 * _B, _R_BOND))
    d = op_union(d, sd_capsule(p, n1, n1 + _T0 * _B, _R_BOND))
    d = op_union(d, sd_capsule(p, n1, n1 + _T3 * _B, _R_BOND))
    d = op_union(d, sd_capsule(p, n2, n2 + _T0 * _B, _R_BOND))
    d = op_union(d, sd_capsule(p, n3, n3 + _T1 * _B, _R_BOND))
    return d


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    # tilt slightly so the tetrahedron reads in 3D
    tb = 0.30
    ct = wp.cos(tb)
    st = wp.sin(tb)
    q = wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])
    return _lattice(q)


@wp.func
def _mat_id(p: wp.vec3, time: float) -> int:
    # bonds thinner than atoms; tint the bonds blue to separate them.
    # recompute atom-only distance to decide atom vs bond
    a = time * 0.4
    ca = wp.cos(a)
    sa = wp.sin(a)
    q = wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])
    tb = 0.30
    ct = wp.cos(tb)
    st = wp.sin(tb)
    q = wp.vec3(q[0], ct * q[1] - st * q[2], st * q[1] + ct * q[2])
    da = _atom(q, wp.vec3(0.0, 0.0, 0.0), _R_ATOM)
    da = op_union(da, _atom(q, _T0 * _B, _R_ATOM))
    da = op_union(da, _atom(q, _T1 * _B, _R_ATOM))
    da = op_union(da, _atom(q, _T2 * _B, _R_ATOM))
    da = op_union(da, _atom(q, _T3 * _B, _R_ATOM))
    if da < 0.02:
        return 0    # silicon atom
    return 10       # tint-blue bond


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
    for _ in range(150):
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
    m = _mat_id(p, time)
    ao = _ao(p, n, time)
    emit = wp.vec3(0.0, 0.0, 0.0)
    if m == 10:
        emit = wp.vec3(0.05, 0.10, 0.28)   # faint glow along the covalent bonds
    img[i, j] = ec.lit(n, rd, m, ao, emit)


def _render(width, height, time, mouse, device):
    az = 0.5 + float(mouse[0]) * 0.01
    el = 0.20 + float(mouse[1]) * 0.005
    dist = 6.2
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.2,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
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
    name="silicon_crystal",
    description="the diamond-cubic silicon lattice — a central atom covalently "
                "bonded to four tetrahedral neighbours (the sp3 bonds), the crystal "
                "carrying on outward. Bonds glow faintly blue. Slowly rotating.",
    renderer=_render,
)
