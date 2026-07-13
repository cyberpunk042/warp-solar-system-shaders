"""NVMe SSD — an M.2 solid-state drive: flash storage on a stick.

An SSD has no moving parts: it stores data as trapped charge in NAND flash cells
(the `nand_flash_cell` from the components round), tiled billions-deep into a few
big flash packages. A controller chip manages them and talks NVMe over the PCIe
edge connector; a small DRAM chip caches the mapping tables. This is the tiny M.2
"2280" form factor — 22 mm wide, 80 mm long, a key notch and a screw-mount
half-circle at the far end. Shown from above. See
``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 45.0
_BX = 2.7        # board half-length
_BY = 0.06       # board half-thickness (flat)
_BZ = 0.62       # board half-width


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.15 * wp.sin(time * 0.5)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    board = sd_box(q, wp.vec3(_BX, _BY, _BZ)) - 0.01
    # screw-mount half-circle notch at the +x end
    mount = sd_cylinder(wp.vec3(q[0] - _BX, q[2], q[1]), 0.3, 0.16)
    board = op_subtract(board, mount)
    # M-key notch in the edge connector at the -x end (off to +z)
    key = sd_box(q - wp.vec3(-_BX, 0.0, 0.42), wp.vec3(0.14, 0.2, 0.22))
    return op_subtract(board, key)


@wp.func
def _chips(q: wp.vec3) -> float:
    ctrl = sd_box(q - wp.vec3(0.4, _BY + 0.07, 0.0), wp.vec3(0.34, 0.08, 0.34)) - 0.01
    nand0 = sd_box(q - wp.vec3(1.55, _BY + 0.06, 0.0), wp.vec3(0.42, 0.07, 0.46)) - 0.01
    nand1 = sd_box(q - wp.vec3(-0.75, _BY + 0.06, 0.0), wp.vec3(0.42, 0.07, 0.46)) - 0.01
    dram = sd_box(q - wp.vec3(-1.7, _BY + 0.05, 0.28), wp.vec3(0.28, 0.06, 0.2)) - 0.01
    return wp.min(wp.min(ctrl, nand0), wp.min(nand1, dram))


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    return op_union(_pcb(q), _chips(q))


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
    if _chips(q) <= _pcb(q):
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))     # black chips
        return
    # PCB: gold edge fingers on the -x end top face, else green mask
    if q[0] < (-_BX + 0.5) and q[1] > (_BY - 0.02):
        fz = q[2] / 0.11 - wp.floor(q[2] / 0.11)
        if fz > 0.3:
            img[i, j] = ec.lit(n, rd, 2, ao, wp.vec3(0.0, 0.0, 0.0))
        else:
            img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))
    else:
        img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    az = 0.25 + float(mouse[0]) * 0.01
    el = 0.66 + float(mouse[1]) * 0.005
    dist = 6.6
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
    return ec.finish(img.numpy(), width, height)


SCENE = Scene(
    name="nvme_ssd",
    description="an M.2 NVMe solid-state drive — a controller chip, big NAND flash "
                "packages, and a DRAM cache on a slim green PCB with a gold PCIe edge "
                "connector and a screw-mount notch. Flash storage, no moving parts.",
    renderer=_render,
)
