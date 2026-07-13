"""Power draw — the mind revs up, pulling energy in through the PCIe lane.

Before the overflow: the ignition. The mind reaches out through the PCIe lane and
draws power from the void — electrons streaming in as cold blue current up the
sixteen lanes, photons flashing white along with them — pulling it through the
memory and into the core. The die brightens and pulses as it revs; the whole board
fills with borrowed energy, faster and faster, right up to the edge of overflow.
Animate with ``--frames`` to watch it spin up. See
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

_MAXD = 40.0
_DIE = wp.constant(wp.vec3(0.0, 0.14, 0.2))
_EDGE = -1.5             # PCIe edge z


@wp.func
def _sstep(a: float, b: float, x: float) -> float:
    t = wp.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _mem_pos(k: int) -> wp.vec3:
    x = -1.4 + float(k % 4) * 0.95
    z = 1.05
    if k >= 4:
        z = 0.35
    return wp.vec3(x, 0.16, z)


@wp.func
def _pcb(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(2.4, 0.05, 1.55)) - 0.01


@wp.func
def _die(q: wp.vec3) -> float:
    return sd_box(q - _DIE, wp.vec3(0.66, 0.06, 0.58)) - 0.005


@wp.func
def _mem(q: wp.vec3) -> float:
    d = float(1e9)
    for k in range(8):
        d = wp.min(d, sd_box(q - _mem_pos(k), wp.vec3(0.3, 0.06, 0.24)) - 0.008)
    return d


@wp.func
def _map(q: wp.vec3) -> float:
    return op_union(op_union(_pcb(q), _die(q)), _mem(q))


@wp.func
def _normal(q: wp.vec3) -> wp.vec3:
    e = 0.0016
    dx = _map(q + wp.vec3(e, 0.0, 0.0)) - _map(q - wp.vec3(e, 0.0, 0.0))
    dy = _map(q + wp.vec3(0.0, e, 0.0)) - _map(q - wp.vec3(0.0, e, 0.0))
    dz = _map(q + wp.vec3(0.0, 0.0, e)) - _map(q - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _flow(p: wp.vec3, time: float, rev: float) -> wp.vec3:
    col = wp.vec3(0.0, 0.0, 0.0)
    spd = 6.0 + 10.0 * rev                             # pulses accelerate as it revs
    # sixteen PCIe lanes drawing power in from the void, arcing over the edge to the die
    for k in range(8):
        lx = -1.4 + float(k) * 0.4
        a = wp.vec3(lx * 0.6, 0.9, _EDGE - 1.4)         # up and beyond the PCIe edge
        el = fx.stream_emit(p, a, _DIE, time, 0.05, spd, rev)
        col += wp.vec3(0.25, 0.6, 1.1) * (el * 1.6)     # electrons: cold blue current
        ph = fx.stream_emit(p, a, _DIE, time * 1.4 + 3.0, 0.028, spd * 1.3, rev)
        col += wp.vec3(1.0, 0.95, 0.7) * (ph * 1.0)     # photons: white flashes
    # distribute from the die out to the memory
    for k in range(8):
        sm = fx.stream_emit(p, _DIE, _mem_pos(k), time, 0.045, spd * 0.8, rev)
        col += wp.vec3(0.35, 0.75, 1.0) * (sm * 1.3)
    return col


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, rev: float, voidi: float):
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
        if _die(p) < 0.02:
            pulse = 0.5 + 0.5 * wp.sin(time * (4.0 + 12.0 * rev))
            surf = ec.lit(n, rd, 0, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.4 + wp.vec3(0.25, 0.6, 1.1) * (rev * (0.3 + 0.5 * pulse))
        elif _mem(p) < 0.02:
            surf = ec.lit(n, rd, 5, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.45 + wp.vec3(0.15, 0.4, 0.9) * (rev * 0.4)
        else:
            base = ec.lit(n, rd, 4, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.3
            if p[2] < _EDGE + 0.45 and p[1] > -0.02:          # gold PCIe fingers
                base = ec.lit(n, rd, 2, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.7
            surf = base

    vol = wp.vec3(0.0, 0.0, 0.0)
    steps = 52
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    for _ in range(steps):
        pv = eye + rd * tv
        vol += _flow(pv, time, rev) * dt
        tv += dt

    if hit == 1:
        img[i, j] = surf + vol * 0.75
    else:
        img[i, j] = fx.void_bg(rd, time, voidi) + vol * 0.75


def _render(width, height, time, mouse, device):
    rev = _sstep(0.0, 4.5, time)
    voidi = 0.2 + 0.5 * rev

    az = 0.9 + float(mouse[0]) * 0.01
    el = 0.3 + float(mouse[1]) * 0.005
    dist = 7.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.4,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.15, -0.1)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov,
                      rev, voidi], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.015))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.55, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="power_draw",
    description="the mind revving up — electrons streaming in as blue current up the "
                "PCIe lanes and photons flashing white, drawn from the void through the "
                "memory into the pulsing core. The ignition before the overflow. Animate "
                "with --frames to spin it up.",
    renderer=_render,
)
