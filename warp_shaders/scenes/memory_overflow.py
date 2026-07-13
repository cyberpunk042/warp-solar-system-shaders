"""Memory overflow — one block fills, overheats, and detonates through its roof.

Close on a single stacked memory block. The mind pours borrowed energy in and it
fills **layer by layer**, each stratum lighting hotter than the last — red, orange,
white — as the charge refills faster than it can drain. At the top the charge has
nowhere to go: it pinches into a singularity and the block detonates, a plasma column
punching straight up through the roof of the package on a shockwave. A mini atomic
bomb, one per memory block. Animate with ``--frames``. See
``docs/research/37-gpu-singularity.md``.
"""

import math

import numpy as np
import warp as wp

from ..procedural.sdf import op_union, sd_box
from .. import electronics_common as ec
from .. import gpu_fx as fx
from ..engine import post
from ..scene import Scene

_MAXD = 30.0
_BLK = wp.constant(wp.vec3(0.72, 0.46, 0.56))   # block half-extents
_TOP = 0.46


@wp.func
def _sstep(a: float, b: float, x: float) -> float:
    t = wp.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _slab(q: wp.vec3) -> float:
    return sd_box(q - wp.vec3(0.0, -0.62, 0.0), wp.vec3(1.7, 0.12, 1.35)) - 0.01


@wp.func
def _block(q: wp.vec3) -> float:
    return sd_box(q, _BLK) - 0.01


@wp.func
def _map(q: wp.vec3) -> float:
    return op_union(_slab(q), _block(q))


@wp.func
def _normal(q: wp.vec3) -> wp.vec3:
    e = 0.0016
    dx = _map(q + wp.vec3(e, 0.0, 0.0)) - _map(q - wp.vec3(e, 0.0, 0.0))
    dy = _map(q + wp.vec3(0.0, e, 0.0)) - _map(q - wp.vec3(0.0, e, 0.0))
    dz = _map(q + wp.vec3(0.0, 0.0, e)) - _map(q - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, fill: float, heat: float,
                   sing: float, bl: float, voidi: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(130):
        p = eye + rd * t
        d = _map(p)
        if d < 0.0009 * t + 0.0005:
            hit = 1
            break
        t += d * 0.9
        if t > _MAXD:
            break

    t_end = _MAXD
    if hit == 1:
        t_end = t

    surf = wp.vec3(0.0, 0.0, 0.0)
    if hit == 1:
        p = eye + rd * t
        n = _normal(p)
        if _block(p) < 0.02:
            # stacked-memory block: horizontal layers light up as the charge fills
            ny = (p[1] + _TOP) / (2.0 * _TOP)                     # 0 bottom -> 1 top
            lay = ny * 9.0
            edge = lay - wp.floor(lay)
            filled = float(0.0)
            if ny < fill:
                filled = 1.0
            base = ec.lit(n, rd, 5, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.35
            layerhot = wp.clamp(heat * (0.4 + 0.6 * (1.0 - ny)), 0.0, 1.0)  # hotter lower/earlier
            glow = fx.heat_color(layerhot) * (0.25 + 0.9 * filled)
            if edge < 0.12:
                glow = glow * 0.5                                 # dark seams between layers
            surf = base + glow
            if bl > 0.0 and p[1] > _TOP - 0.06:
                surf = surf * wp.clamp(1.0 - bl * 3.0, 0.0, 1.0)  # roof blown open
        else:
            surf = ec.lit(n, rd, 4, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.3

    # volumetrics: singularity pinch above the block + the detonation column
    vol = wp.vec3(0.0, 0.0, 0.0)
    steps = 60
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    sc = wp.vec3(0.0, _TOP + 0.25 + sing * 0.5, 0.0)
    for _ in range(steps):
        pv = eye + rd * tv
        se = fx.singularity_emit(pv, sc, time, sing)
        vol += wp.vec3(0.75, 0.5, 1.0) * (se * dt)
        be = fx.blast_emit(pv, wp.vec3(0.0, _TOP - 0.05, 0.0), bl, 3.4)
        hot = wp.clamp(1.1 - bl, 0.0, 1.0)
        vol += fx.heat_color(0.45 + 0.55 * hot) * (be * dt)
        tv += dt

    if hit == 1:
        img[i, j] = surf + vol * 0.6
    else:
        img[i, j] = fx.void_bg(rd, time, voidi) + vol * 0.6


def _render(width, height, time, mouse, device):
    fill = _sstep(0.0, 2.6, time)
    heat = _sstep(0.8, 3.2, time)
    sing = _sstep(2.9, 3.5, time) * _sstep(4.8, 3.7, time)
    bl = (time - 3.6) / 1.4
    voidi = _sstep(3.2, 5.5, time)

    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.24 + float(mouse[1]) * 0.005
    dist = 4.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.5,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.35, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(48.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov,
                      fill, heat, sing, bl, voidi], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.15, strength=0.5, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="memory_overflow",
    description="one memory block filling with charge layer by layer, overheating "
                "red to white, then detonating — a singularity pinch and a plasma column "
                "punching up through the roof of the package. The mini atomic bomb, per "
                "block, up close. Animate with --frames.",
    renderer=_render,
)
