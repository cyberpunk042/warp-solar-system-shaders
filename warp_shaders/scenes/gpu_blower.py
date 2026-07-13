"""GPU (blower / induction fan) — a single centrifugal fan that ducts air out the back.

A blower cooler works differently from the open-air card: one **centrifugal
(squirrel-cage) fan** at the far end pulls air in through a round grille and forces
it sideways through a sealed finned tunnel, so all the hot air is *induced* along
the card and exhausted straight out of the case through the bracket vents — instead
of spilling it inside. It is louder but self-contained, which is why blower cards
suit small cases and dense multi-GPU servers. Sleek closed shroud, ringed intake
grille, louvered exhaust. See ``docs/research/36-boards-and-memory-blocks.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_subtract, op_union, sd_box, sd_cylinder
from .. import electronics_common as ec
from ..scene import Scene

_MAXD = 60.0
_BLOW = 2.35      # blower centre x
_BR = 0.95        # blower radius
_TOP = 0.62


@wp.func
def _rot(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.4 + 0.08 * wp.sin(time * 0.35)
    ca = wp.cos(a)
    sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _pcb(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.14, 0.06), wp.vec3(3.5, 0.05, 1.15)) - 0.01


@wp.func
def _shroud(q: wp.vec3) -> float:
    body = sd_box(q - wp.vec3(0.0, 0.3, 0.0), wp.vec3(3.45, 0.32, 1.12)) - 0.04
    well = sd_cylinder(q - wp.vec3(_BLOW, 0.66, 0.0), 0.1, _BR)
    return op_subtract(body, well)


@wp.func
def _hub(q: wp.vec3) -> float:
    return sd_cylinder(q - wp.vec3(_BLOW, 0.55, 0.0), 0.13, 0.16)


@wp.func
def _bracket(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(-3.6, 0.24, 0.0), wp.vec3(0.05, 0.6, 1.16)) - 0.01


@wp.func
def _power(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(-1.2, 0.66, -0.8), wp.vec3(0.44, 0.16, 0.22)) - 0.01


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    q = _rot(p, time)
    d = op_union(_pcb(q), _shroud(q))
    d = op_union(d, _hub(q))
    d = op_union(d, _bracket(q))
    return op_union(d, _power(q))


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0014
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


@wp.func
def _blower(dx: float, dz: float, time: float) -> float:
    """A centrifugal-fan look through a ringed intake grille."""
    r = wp.sqrt(dx * dx + dz * dz) / _BR
    if r < 0.16:
        return 0.28                                       # hub
    if r > 0.92:
        return 0.5                                        # frame
    ang = wp.atan2(dz, dx) + time * 4.0
    blades = 0.5 + 0.5 * wp.sin(ang * 46.0)               # many fine squirrel-cage blades
    shade = 0.14 + 0.32 * blades
    ring = r * 7.0 - wp.floor(r * 7.0)
    if ring < 0.16:
        shade = 0.42                                      # concentric grille ring
    return shade


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
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.86
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, time)
    ao = _ao(p, n, time)

    q = _rot(p, time)
    dp = _pcb(q)
    dsh = _shroud(q)
    dhub = _hub(q)
    dbr = _bracket(q)
    dpw = _power(q)
    mind = wp.min(wp.min(wp.min(dp, dsh), wp.min(dhub, dbr)), dpw)
    eps = 0.0012
    if dbr <= mind + eps:
        col = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0))
        # louvered exhaust vents on the bracket
        lo = q[1] / 0.13 - wp.floor(q[1] / 0.13)
        if lo < 0.55 and wp.abs(n[0]) > 0.5:
            col = col * 0.25
        img[i, j] = col
    elif dpw <= mind + eps:
        img[i, j] = ec.lit(n, rd, 5, ao, wp.vec3(0.0, 0.0, 0.0))
    elif dhub <= mind + eps:
        img[i, j] = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0)) * (0.3 + 0.6 * _blower(q[0] - _BLOW, q[2], time))
    elif dsh <= mind + eps:
        # premium gunmetal shroud
        base = ec.lit(n, rd, 7, ao, wp.vec3(0.0, 0.0, 0.0)) * 0.5
        r = wp.sqrt((q[0] - _BLOW) * (q[0] - _BLOW) + q[2] * q[2])
        if q[1] > 0.5 and n[1] > 0.55 and r < _BR:
            s = _blower(q[0] - _BLOW, q[2], time)
            img[i, j] = wp.vec3(s, s, s * 1.08) * (0.5 + 0.4 * ao)
        elif q[1] > 0.55 and n[1] > 0.55:
            # top: brushed sheen + a thin cyan accent line down the spine
            out = base + wp.vec3(0.05, 0.06, 0.08)
            if wp.abs(q[2]) < 0.04:
                out = out + wp.vec3(0.2, 0.55, 1.0) * 0.55
            img[i, j] = out
        else:
            img[i, j] = base
    else:
        img[i, j] = ec.lit(n, rd, 4, ao, wp.vec3(0.0, 0.0, 0.0))   # PCB edge


def _render(width, height, time, mouse, device):
    az = 0.5 + float(mouse[0]) * 0.01
    el = 0.46 + float(mouse[1]) * 0.005
    dist = 9.4
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.5,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.12, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    return ec.finish(img.numpy(), width, height, threshold=1.7, strength=0.32)


SCENE = Scene(
    name="gpu_blower",
    description="a blower-style graphics card — a single centrifugal (squirrel-cage) "
                "fan under a ringed intake grille that ducts all the hot air out the "
                "louvered bracket, in a sleek closed gunmetal shroud. The self-contained "
                "cooler for small cases and servers.",
    renderer=_render,
)
