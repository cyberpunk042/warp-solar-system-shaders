"""Spark gap — an arc jumping between two electrodes.

The simplest breakdown: two conductor knobs held a small distance apart, the voltage
across them ramped until the air ionises and a hot **arc** jumps the gap — a short,
fierce, branching channel that flickers and re-strikes. Animate with ``--frames``. See
``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..procedural.sdf import op_union, sd_cylinder, sd_sphere
from ..scene import Scene

_MAXD = 30.0
_A = wp.constant(wp.vec3(-0.9, 0.0, 0.0))     # left electrode tip
_B = wp.constant(wp.vec3(0.9, 0.0, 0.0))      # right electrode tip


@wp.func
def _map(p: wp.vec3) -> float:
    la = sd_sphere(p - _A, 0.28)
    lb = sd_sphere(p - _B, 0.28)
    ra = sd_cylinder(wp.vec3(p[1], p[0] + 1.7, p[2]) - wp.vec3(0.0, 0.0, 0.0), 0.6, 0.12)  # stalk L
    rb = sd_cylinder(wp.vec3(p[1], -p[0] + 1.7, p[2]), 0.6, 0.12)                            # stalk R
    return op_union(op_union(la, lb), op_union(ra, rb))


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
                   width: int, height: int, tanfov: float, flash: float, width_b: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(140):
        p = eye + rd * t
        d = _map(p)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    up_ = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    col = wp.vec3(0.015, 0.016, 0.028) * (1.0 - up_) + wp.vec3(0.03, 0.03, 0.05) * up_
    if hit == 1:
        p = eye + rd * t
        n = _normal(p)
        ld = wp.normalize(wp.vec3(0.0, 0.0, 1.0) - p)
        diff = wp.max(wp.dot(n, ld), 0.0)
        h = wp.normalize(ld - rd)
        spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 48.0)
        arclit = flash * wp.exp(-wp.length(p) * 0.6)                # arc lights the knobs
        col = wp.vec3(0.5, 0.52, 0.58) * (0.1 + 0.5 * diff + 1.4 * arclit) + wp.vec3(1.0) * (spec * (0.4 + flash))

    g = float(0.0)
    core = float(0.0)
    for k in range(npts):
        g += el.pt_glow(eye, rd, pts[k], width_b)
        core += el.pt_glow(eye, rd, pts[k], width_b * 0.4)
    col += wp.vec3(0.5, 0.65, 1.0) * (wp.clamp(g, 0.0, 3.0) * flash * 1.7)
    col += wp.vec3(1.0, 0.95, 1.0) * (wp.clamp(core, 0.0, 4.0) * flash * 2.4)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    flash = 0.4 + 0.6 * abs(math.sin(time * 11.0)) * (0.5 + 0.5 * math.sin(time * 53.0))
    frame = int(math.floor(time * 30.0))
    pts = el.generate_bolt((-0.62, 0.0, 0.0), (0.62, 0.0, 0.0), seed=frame, gens=4,
                           jitter=0.32, branch_prob=0.35, pts_per_seg=4)
    parr, npts = el.upload_points(pts, device)

    az = 0.15 + math.sin(time * 0.1) * 0.1 + float(mouse[0]) * 0.01
    el_ang = 0.12 + float(mouse[1]) * 0.005
    dist = 4.6
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), dist * math.sin(el_ang) + 0.2,
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, parr, npts, eye, fwd, right, up, width, height, tanfov,
                      float(flash), 0.05], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.55, radius=r, passes=4, octaves=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="spark_gap",
    description="an electric arc jumping between two conductor knobs — the air breaking "
                "down into a hot, forked, flickering channel across the gap, lighting the "
                "electrodes. The simplest spark. Animate with --frames.",
    renderer=_render,
)
