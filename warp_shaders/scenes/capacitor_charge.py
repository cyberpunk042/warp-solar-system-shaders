"""Capacitor charge — two plates charging, storing the field, then discharging.

Current rushes in along the wires and piles opposite charge onto two parallel plates;
the **electric field** builds in the gap between them (glowing brighter as Q = CV rises,
on the RC curve). At full charge the gap breaks down and it **dumps** the stored energy
in a snapping arc — then it charges again. Animate with ``--frames``. See
``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..procedural.sdf import op_union, sd_box
from ..scene import Scene

_MAXD = 30.0
_PX = 0.62          # plate x offset


@wp.func
def _map(p: wp.vec3) -> float:
    pa = sd_box(p - wp.vec3(-_PX, 0.4, 0.0), wp.vec3(0.05, 0.85, 0.85)) - 0.01
    pb = sd_box(p - wp.vec3(_PX, 0.4, 0.0), wp.vec3(0.05, 0.85, 0.85)) - 0.01
    wa = sd_box(p - wp.vec3(-_PX, -0.75, 0.0), wp.vec3(0.05, 0.55, 0.05))
    wb = sd_box(p - wp.vec3(_PX, -0.75, 0.0), wp.vec3(0.05, 0.55, 0.05))
    base = sd_box(p - wp.vec3(0.0, -1.3, 0.0), wp.vec3(1.2, 0.08, 0.4)) - 0.02
    return op_union(op_union(pa, pb), op_union(op_union(wa, wb), base))


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0016
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), pts: wp.array(dtype=wp.vec3), npts: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
                   width: int, height: int, tanfov: float, time: float, charge: float,
                   flash: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(150):
        p = eye + rd * t
        d = _map(p)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    up_ = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.014, 0.016, 0.026) * (1.0 - up_) + wp.vec3(0.028, 0.03, 0.05) * up_
    if hit == 1:
        p = eye + rd * t
        n = _normal(p)
        ld = wp.normalize(wp.vec3(0.4, 0.6, 1.0))
        diff = wp.max(wp.dot(n, ld), 0.0)
        base = wp.vec3(0.5, 0.34, 0.16)                          # copper
        if p[1] > -0.5 and wp.abs(wp.abs(p[0]) - _PX) < 0.1:
            side = wp.vec3(0.9, 0.25, 0.2)                        # + plate glows red
            if p[0] > 0.0:
                side = wp.vec3(0.2, 0.4, 0.95)                    # - plate glows blue
            base = base * 0.4 + side * (0.2 + 0.9 * charge)
        col = base * (0.12 + 0.7 * diff)

    g = float(0.0)
    core = float(0.0)
    for k in range(npts):
        g += el.pt_glow(eye, rd, pts[k], 0.05)
        core += el.pt_glow(eye, rd, pts[k], 0.02)
    col += wp.vec3(0.7, 0.8, 1.0) * (wp.clamp(g, 0.0, 3.0) * (0.15 + 0.5 * charge + flash * 1.6))
    col += wp.vec3(1.0, 0.95, 1.0) * (wp.clamp(core, 0.0, 4.0) * flash * 2.2)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    cyc = time - 3.0 * math.floor(time / 3.0)
    if cyc < 2.4:
        charge = 1.0 - math.exp(-cyc / 0.6)                       # RC charge
        flash = 0.0
    else:
        charge = math.exp(-(cyc - 2.4) / 0.15)                    # dumps
        flash = math.exp(-(cyc - 2.4) * 7.0) * (0.6 + 0.4 * math.sin(cyc * 60.0))

    frame = int(math.floor(time * 24.0))
    rng = np.random.RandomState((frame * 2654435761) & 0x7FFFFFFF)
    allpts = []
    for s in range(7):
        y = 0.4 + rng.uniform(-0.7, 0.7)
        z = rng.uniform(-0.7, 0.7)
        seg = np.linspace([-_PX + 0.06, y, z], [_PX - 0.06, y, z], 6)
        allpts.append(seg.astype(np.float32))
    if flash > 0.02:
        b = el.generate_bolt((-_PX + 0.06, 0.4, 0.0), (_PX - 0.06, 0.4, 0.0), seed=frame,
                             gens=4, jitter=0.28, branch_prob=0.3, pts_per_seg=4)
        allpts.append(b)
    pts = np.concatenate(allpts, axis=0)
    parr, npts = el.upload_points(pts, device)

    az = 0.7 + math.sin(time * 0.1) * 0.15 + float(mouse[0]) * 0.01
    el_ang = 0.16 + float(mouse[1]) * 0.005
    dist = 5.4
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), dist * math.sin(el_ang) + 0.3,
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 0.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, parr, npts, eye, fwd, right, up, width, height, tanfov,
                      time, float(charge), float(flash)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.018))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.5, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="capacitor_charge",
    description="a parallel-plate capacitor charging on the RC curve — current piling "
                "opposite charge on the plates (red + / blue -), the field glowing brighter "
                "in the gap as Q=CV rises, then dumping in a snapping arc. Animate with --frames.",
    renderer=_render,
)
