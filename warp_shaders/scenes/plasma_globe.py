"""Plasma globe — filaments reaching from the core electrode to the glass.

The novelty-shop sphere: a small electrode at the centre sits at high-frequency high
voltage; the thin low-pressure gas ionises into snaking violet-pink **filaments** that
reach out to the inner glass, crowding toward wherever a hand touches (a moving bright
spot on the shell). Animate with ``--frames``. See ``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..particles import ray_sphere
from ..scene import Scene

_R = 2.0            # globe radius


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), pts: wp.array(dtype=wp.vec3), npts: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
                   width: int, height: int, tanfov: float, time: float, touch: wp.vec3):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    col = wp.vec3(0.01, 0.01, 0.02)
    # filaments (dense glowing points) — additive, seen through the glass
    g = float(0.0)
    core = float(0.0)
    for k in range(npts):
        g += el.pt_glow(eye, rd, pts[k], 0.055)
        core += el.pt_glow(eye, rd, pts[k], 0.022)
    col += wp.vec3(0.8, 0.35, 1.0) * (wp.clamp(g, 0.0, 3.0) * 1.2)
    col += wp.vec3(1.0, 0.85, 1.0) * (wp.clamp(core, 0.0, 4.0) * 1.6)

    # central electrode bulb + the touch hot-spot on the shell
    col += wp.vec3(1.0, 0.7, 0.9) * (el.corona(eye, rd, wp.vec3(0.0, 0.0, 0.0), 0.28) * 1.5)
    col += wp.vec3(1.0, 0.6, 0.9) * (el.corona(eye, rd, touch, 0.5) * 0.8)

    # glass shell: a faint Fresnel rim + a violet tint where the ray grazes the sphere
    tn, tf, hitg = ray_sphere(eye, rd, wp.vec3(0.0, 0.0, 0.0), _R)
    if hitg == 1 and tn > 0.0:
        hitp = eye + rd * tn
        n = wp.normalize(hitp)
        fres = wp.pow(1.0 - wp.max(-wp.dot(rd, n), 0.0), 3.0)
        col += wp.vec3(0.3, 0.2, 0.5) * (fres * 0.5)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    frame = int(math.floor(time * 20.0))
    rng = np.random.RandomState((frame * 2654435761) & 0x7FFFFFFF)
    # the touch point drifts around the shell
    ta = time * 0.7
    tb = 0.6 + 0.3 * math.sin(time * 0.5)
    touch = wp.vec3(_R * math.cos(ta) * math.sin(tb),
                    _R * math.cos(tb),
                    _R * math.sin(ta) * math.sin(tb))
    tpt = np.array([float(touch[0]), float(touch[1]), float(touch[2])])

    allpts = []
    for s in range(7):
        # each filament runs from the core to a point on the shell, biased toward the touch
        rd = rng.randn(3)
        rd = rd / (np.linalg.norm(rd) + 1e-6)
        endpt = rd * _R
        if rng.uniform() < 0.5:                      # half crowd toward the touch point
            endpt = 0.4 * endpt + 0.6 * tpt
            endpt = endpt / (np.linalg.norm(endpt) + 1e-6) * _R
        b = el.generate_bolt((0.0, 0.0, 0.0), tuple(endpt), seed=frame * 13 + s, gens=4,
                             jitter=0.42, branch_prob=0.3, pts_per_seg=4)
        allpts.append(b)
    pts = np.concatenate(allpts, axis=0)
    parr, npts = el.upload_points(pts, device)

    az = 0.6 + time * 0.05 + float(mouse[0]) * 0.01
    el_ang = 0.12 + float(mouse[1]) * 0.005
    dist = 6.4
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az), dist * math.sin(el_ang),
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, parr, npts, eye, fwd, right, up, width, height, tanfov,
                      time, touch], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.85, strength=0.55, radius=r, passes=4, octaves=5)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="plasma_globe",
    description="a plasma globe — snaking violet-pink filaments reaching from the central "
                "electrode out to the glass shell, crowding toward a drifting touch point. "
                "Animate with --frames.",
    renderer=_render,
)
