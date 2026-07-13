"""Power grid — current pulsing down the transmission lines at night.

The grid at scale: lattice **pylons** carry sagging catenary conductors across the dark,
and packets of **current** race along the lines from a glowing substation toward the city
on the horizon. The wires glow a faint blue; the moving pulses are bright — the visible
flow of power across the land. Animate with ``--frames``. See
``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..procedural.sdf import op_union, sd_box
from ..scene import Scene

_MAXD = 80.0
_SPAN = 4.5             # pylon spacing along x
_HEIGHTS = [2.6, 3.1, 3.6]


@wp.func
def _pylon(p: wp.vec3, cx: float) -> float:
    q = p - wp.vec3(cx, 0.0, 0.0)
    # two tapered legs + a mast + three crossarms (a lattice-tower silhouette)
    legL = sd_box(q - wp.vec3(-0.35 + q[1] * 0.06, 1.4, 0.0), wp.vec3(0.06, 1.5, 0.06))
    legR = sd_box(q - wp.vec3(0.35 - q[1] * 0.06, 1.4, 0.0), wp.vec3(0.06, 1.5, 0.06))
    mast = sd_box(q - wp.vec3(0.0, 3.0, 0.0), wp.vec3(0.07, 1.1, 0.07))
    arm0 = sd_box(q - wp.vec3(0.0, 2.6, 0.0), wp.vec3(0.95, 0.05, 0.05))
    arm1 = sd_box(q - wp.vec3(0.0, 3.1, 0.0), wp.vec3(0.75, 0.05, 0.05))
    arm2 = sd_box(q - wp.vec3(0.0, 3.6, 0.0), wp.vec3(0.5, 0.05, 0.05))
    return op_union(op_union(op_union(legL, legR), mast), op_union(op_union(arm0, arm1), arm2))


@wp.func
def _map(p: wp.vec3) -> float:
    d = _pylon(p, -_SPAN)
    d = op_union(d, _pylon(p, 0.0))
    d = op_union(d, _pylon(p, _SPAN))
    ground = p[1] + 0.02
    return wp.min(d, ground)


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.004
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), wire: wp.array(dtype=wp.vec3), nw: int,
                   pulse: wp.array(dtype=wp.vec3), npz: int, eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanfov: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.1)
    hit = int(0)
    for _ in range(160):
        p = eye + rd * t
        d = _map(p)
        if d < 0.002 * t + 0.001:
            hit = 1
            break
        t += d * 0.9
        if t > _MAXD:
            break

    # night sky + a warm city glow on the horizon
    up_ = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    horizon = wp.pow(1.0 - wp.clamp(wp.abs(rd[1]) + 0.02, 0.0, 1.0), 6.0)
    col = wp.vec3(0.02, 0.03, 0.06) * (1.0 - up_) + wp.vec3(0.04, 0.05, 0.09) * up_
    col += wp.vec3(0.5, 0.35, 0.15) * (horizon * 0.5)
    if hit == 1:
        p = eye + rd * t
        n = _normal(p)
        if p[1] < 0.05:
            col = wp.vec3(0.03, 0.035, 0.05) * (0.4 + 0.6 * wp.max(n[1], 0.0)) * wp.exp(-t * 0.02)
        else:
            ld = wp.normalize(wp.vec3(0.3, 0.8, 0.4))
            col = wp.vec3(0.16, 0.17, 0.2) * (0.2 + 0.8 * wp.max(wp.dot(n, ld), 0.0))

    # conductors: a faint continuous blue glow (dense points)
    gw = float(0.0)
    for k in range(nw):
        gw += el.pt_glow(eye, rd, wire[k], 0.03)
    col += wp.vec3(0.3, 0.45, 0.9) * (wp.clamp(gw, 0.0, 2.0) * 0.5)
    # current packets: bright moving pulses
    gp = float(0.0)
    for k in range(npz):
        gp += el.pt_glow(eye, rd, pulse[k], 0.055)
    col += wp.vec3(0.7, 0.9, 1.0) * (wp.clamp(gp, 0.0, 3.0) * 1.6)
    img[i, j] = col


def _catenary(x0, x1, y, sag, z, nseg):
    xs = np.linspace(0.0, 1.0, nseg)
    pts = []
    for u in xs:
        x = x0 + (x1 - x0) * u
        yy = y - sag * 4.0 * u * (1.0 - u)
        pts.append([x, yy, z])
    return pts


def _render(width, height, time, mouse, device):
    wirepts = []
    pulsepts = []
    spans = [(-_SPAN, 0.0), (0.0, _SPAN)]
    for (x0, x1) in spans:
        for h in _HEIGHTS:
            for z in (-0.0,):
                wirepts += _catenary(x0, x1, h, 0.55, z, 26)
    # moving current packets along each span/height
    for (x0, x1) in spans:
        for hi in range(len(_HEIGHTS)):
            h = _HEIGHTS[hi]
            phase = (time * 0.35 + hi * 0.3) % 1.0
            for pk in range(2):
                u = (phase + pk * 0.5) % 1.0
                x = x0 + (x1 - x0) * u
                yy = h - 0.55 * 4.0 * u * (1.0 - u)
                pulsepts.append([x, yy, 0.0])
    warr, nw = el.upload_points(np.asarray(wirepts, dtype=np.float32), device)
    parr, npz = el.upload_points(np.asarray(pulsepts, dtype=np.float32), device)

    az = 0.9 + math.sin(time * 0.05) * 0.1 + float(mouse[0]) * 0.01
    el_ang = 0.12 + float(mouse[1]) * 0.005
    dist = 12.0
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), 1.6 + dist * math.sin(el_ang),
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 2.4, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(50.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, warr, nw, parr, npz, eye, fwd, right, up, width, height, tanfov],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.95, strength=0.5, radius=r, passes=4, octaves=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="power_grid",
    description="the transmission grid at night — lattice pylons carrying sagging catenary "
                "conductors, faint-blue wires with bright packets of current racing along "
                "them across the dark toward a city glow on the horizon. Animate with --frames.",
    renderer=_render,
)
