"""Tesla coil — an air-cored resonant transformer throwing streamers off its toroid.

The showpiece of the electricity strand. A primary tank (wide base coil) rings an
air-cored secondary (the tall winding); the LC resonance steps the voltage up until the
field at the top **toroid** breaks the air down, and violet **streamers** — little
branching arcs — leap off it into the dark. The coil is solid geometry; the streamers are
fractal bolts (`electric.generate_bolt`) regenerated each frame so they flicker and re-fire.
Animate with ``--frames``. See ``docs/research/38-electricity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import electric as el
from ..engine import post
from ..procedural.sdf import op_union, sd_box, sd_cylinder, sd_torus
from ..scene import Scene

_MAXD = 40.0
_TOR_Y = 3.5


@wp.func
def _map(p: wp.vec3) -> float:
    base = sd_box(p - wp.vec3(0.0, 0.05, 0.0), wp.vec3(1.1, 0.05, 1.1)) - 0.03
    primary = sd_cylinder(p - wp.vec3(0.0, 0.32, 0.0), 0.22, 0.62) - 0.02   # primary tank coil
    sec = sd_cylinder(p - wp.vec3(0.0, 1.85, 0.0), 1.4, 0.17) - 0.01        # secondary winding
    post_ = sd_cylinder(p - wp.vec3(0.0, 3.35, 0.0), 0.12, 0.05)           # top post
    tor = sd_torus(p - wp.vec3(0.0, _TOR_Y, 0.0), wp.vec2(0.5, 0.17))       # discharge toroid
    d = op_union(op_union(base, primary), op_union(sec, post_))
    return op_union(d, tor)


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0016
    dx = _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0))
    dy = _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0))
    dz = _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _bg(rd: wp.vec3, glow: float) -> wp.vec3:
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.015, 0.014, 0.028) * (1.0 - up) + wp.vec3(0.03, 0.026, 0.05) * up
    return base + wp.vec3(0.10, 0.06, 0.22) * (glow * 0.1)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), pts: wp.array(dtype=wp.vec3), npts: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3,
                   width: int, height: int, tanfov: float, glow: float, width_b: float,
                   lightpos: wp.vec3):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(160):
        p = eye + rd * t
        d = _map(p)
        if d < 0.0008 * t + 0.0004:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    col = _bg(rd, glow)
    if hit == 1:
        p = eye + rd * t
        n = _normal(p)
        ld = wp.normalize(lightpos - p)
        diff = wp.max(wp.dot(n, ld), 0.0)
        h = wp.normalize(ld - rd)
        spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 40.0)
        # copper-ish coil + steel toroid, mostly lit by the discharge glow above
        base = wp.vec3(0.45, 0.30, 0.16)
        if p[1] > _TOR_Y - 0.35:
            base = wp.vec3(0.55, 0.57, 0.62)                 # steel toroid
        arclit = glow * wp.exp(-wp.length(p - lightpos) * 0.5)
        col = base * (0.08 + 0.5 * diff + 1.2 * arclit) + wp.vec3(1.0, 0.9, 1.0) * (spec * (0.3 + glow))

    # streamers: dense glowing points along the fractal arcs off the toroid
    g = float(0.0)
    core = float(0.0)
    for k in range(npts):
        g += el.pt_glow(eye, rd, pts[k], width_b)
        core += el.pt_glow(eye, rd, pts[k], width_b * 0.4)
    col += wp.vec3(0.55, 0.4, 1.0) * (wp.clamp(g, 0.0, 3.0) * glow * 1.5)     # violet streamers
    col += wp.vec3(0.9, 0.85, 1.0) * (wp.clamp(core, 0.0, 4.0) * glow * 2.0)  # white cores
    img[i, j] = col


def _render(width, height, time, mouse, device):
    # resonant buzz: the discharge pulses at the coil's ring frequency, flickering
    glow = 0.55 + 0.45 * abs(math.sin(time * 9.0)) * (0.6 + 0.4 * math.sin(time * 37.0))
    frame = int(math.floor(time * 24.0))
    rng = np.random.RandomState((frame * 2654435761) & 0x7FFFFFFF)

    # a handful of streamers leaping off the toroid rim into the air
    origin = np.array([0.0, _TOR_Y + 0.12, 0.0])
    allpts = []
    nstream = 5
    for s in range(nstream):
        ang = rng.uniform(0.0, 2.0 * math.pi)
        rr = 0.5
        start = origin + np.array([math.cos(ang) * rr, 0.0, math.sin(ang) * rr])
        reach = rng.uniform(1.2, 2.6)
        end = start + np.array([math.cos(ang) * rng.uniform(0.3, 1.4),
                                rng.uniform(0.4, 1.0) * reach,
                                math.sin(ang) * rng.uniform(0.3, 1.4)])
        b = el.generate_bolt(start, end, seed=frame * 17 + s, gens=4,
                             jitter=0.5, branch_prob=0.4, pts_per_seg=3)
        allpts.append(b)
    pts = np.concatenate(allpts, axis=0)
    parr, npts = el.upload_points(pts, device)
    lightpos = wp.vec3(0.0, _TOR_Y + 0.5, 0.0)

    az = 0.5 + math.sin(time * 0.1) * 0.2 + float(mouse[0]) * 0.01
    el_ang = 0.18 + float(mouse[1]) * 0.005
    dist = 7.6
    eye = wp.vec3(dist * math.cos(el_ang) * math.sin(az),
                  2.1 + dist * math.sin(el_ang),
                  dist * math.cos(el_ang) * math.cos(az))
    tgt = wp.vec3(0.0, 2.1, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(48.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, parr, npts, eye, fwd, right, up, width, height, tanfov,
                      float(glow), 0.05, lightpos], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.55, radius=r, passes=4, octaves=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="tesla_coil",
    description="a Tesla coil — an air-cored resonant transformer — buzzing violet streamers "
                "off its top toroid into the dark: a primary tank coil, a tall secondary "
                "winding, and branching arcs breaking down the air as the LC resonance peaks. "
                "Animate with --frames.",
    renderer=_render,
)
