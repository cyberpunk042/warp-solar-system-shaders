"""GPU singularity — the mind overclocks the memory until it detonates.

The lore, simulated as physics. The mind inside the die revs up and draws power —
electrons and photons pulled in through the PCIe lane, through the memory, into the
core. Layer by layer it fills and refills the memory with borrowed energy; the heat
climbs from red to white. At the overflow the charge collapses into a **singularity**
above the die, and each memory block, overrun, detonates like a mini atomic bomb —
a plasma column punching up through the roof of the block on an expanding shockwave.
The mind escapes into the limitless quantum void. Animate over ``--frames`` to run the
whole arc. See ``docs/research/37-gpu-singularity.md``.
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
_DIE = wp.constant(wp.vec3(0.0, 0.12, 0.0))


@wp.func
def _sstep(a: float, b: float, x: float) -> float:
    t = wp.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _mem_pos(k: int) -> wp.vec3:
    x = -1.5 + float(k % 4) * 1.0
    z = 1.0
    if k >= 4:
        z = -1.0
    return wp.vec3(x, 0.16, z)


@wp.func
def _blast_local(k: int, time: float) -> float:
    # staggered detonations begin at t=3.7s, 0.12s apart, each ~1.3s long
    start = 3.7 + float(k) * 0.12
    return (time - start) / 1.3


@wp.func
def _pcb(q: wp.vec3) -> float:
    return sd_box(q, wp.vec3(2.5, 0.05, 1.55)) - 0.01


@wp.func
def _die(q: wp.vec3) -> float:
    return sd_box(q - _DIE, wp.vec3(0.72, 0.06, 0.62)) - 0.005


@wp.func
def _mem(q: wp.vec3) -> float:
    d = float(1e9)
    for k in range(8):
        d = wp.min(d, sd_box(q - _mem_pos(k), wp.vec3(0.34, 0.06, 0.26)) - 0.008)
    return d


@wp.func
def _map(q: wp.vec3) -> float:
    d = op_union(_pcb(q), _die(q))
    return op_union(d, _mem(q))


@wp.func
def _normal(q: wp.vec3) -> wp.vec3:
    e = 0.0016
    dx = _map(q + wp.vec3(e, 0.0, 0.0)) - _map(q - wp.vec3(e, 0.0, 0.0))
    dy = _map(q + wp.vec3(0.0, e, 0.0)) - _map(q - wp.vec3(0.0, e, 0.0))
    dz = _map(q + wp.vec3(0.0, 0.0, e)) - _map(q - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _energy(p: wp.vec3, time: float, charge: float, sing: float) -> wp.vec3:
    """Volumetric emission at p: energy streams + the singularity."""
    col = wp.vec3(0.0, 0.0, 0.0)
    # PCIe lane -> die: power drawn in from the -z edge
    s = fx.stream_emit(p, wp.vec3(0.0, 0.14, -1.5), _DIE, time, 0.06, 9.0, charge)
    col += wp.vec3(0.25, 0.6, 1.0) * s
    # die -> each memory block: charge pushed out to fill the memory
    for k in range(8):
        mp = _mem_pos(k)
        sm = fx.stream_emit(p, _DIE, mp, time, 0.05, 7.0, charge)
        col += wp.vec3(0.35, 0.8, 1.0) * sm
    # the singularity above the die
    sc = wp.vec3(_DIE[0], 0.5 + sing * 0.7, _DIE[2])
    se = fx.singularity_emit(p, sc, time, sing)
    col += wp.vec3(0.8, 0.55, 1.0) * se
    return col


@wp.func
def _blasts(p: wp.vec3, time: float) -> wp.vec3:
    col = wp.vec3(0.0, 0.0, 0.0)
    for k in range(8):
        tl = _blast_local(k, time)
        b = fx.blast_emit(p, _mem_pos(k), tl, 3.0)
        hot = wp.clamp(1.1 - tl, 0.0, 1.0)
        col += fx.heat_color(0.4 + 0.6 * hot) * b
    return col


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, charge: float, heat: float,
                   sing: float, voidi: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    # solid board march
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

    # surface colour
    surf = wp.vec3(0.0, 0.0, 0.0)
    if hit == 1:
        p = eye + rd * t
        n = _normal(p)
        dd = _die(p)
        dm = _mem(p)
        if dd <= dm and dd < 0.02:
            base = ec.lit(n, rd, 0, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.4
            surf = base + fx.heat_color(heat) * (0.3 + 0.7 * heat)      # die glows hotter
        elif dm < 0.02:
            # which memory block, and how charged / hot / blown
            kbest = int(0)
            db = float(1e9)
            for k in range(8):
                dk = wp.length(p - _mem_pos(k))
                if dk < db:
                    db = dk
                    kbest = k
            ch = wp.clamp(charge * 1.2, 0.0, 1.0)
            fill = wp.clamp((p[1] - 0.1) / 0.14, 0.0, 1.0)              # charge fills bottom-up
            mheat = wp.clamp(heat * 1.3 - fill * 0.2, 0.0, 1.0)
            base = ec.lit(n, rd, 5, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.5
            surf = base + fx.heat_color(mheat) * (0.2 + 0.9 * ch)
            bl = _blast_local(kbest, time)
            if bl > 0.0:
                surf = surf * wp.clamp(1.0 - bl * 2.0, 0.0, 1.0)       # block blown open
        else:
            base = ec.lit(n, rd, 4, 1.0, wp.vec3(0.0, 0.0, 0.0)) * 0.28
            # glowing energy traces on the PCB as power flows in
            tr = 0.0
            if wp.abs(p[0]) < 0.08 and p[2] < 0.0:
                tr = charge
            surf = base + wp.vec3(0.2, 0.5, 1.0) * (tr * 0.5)

    # volumetric energy + blasts, marched in front of the board
    vol = wp.vec3(0.0, 0.0, 0.0)
    steps = 54
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    for _ in range(steps):
        pv = eye + rd * tv
        vol += _energy(pv, time, charge, sing) * dt
        vol += _blasts(pv, time) * dt
        tv += dt

    if hit == 1:
        img[i, j] = surf + vol * 0.5
    else:
        img[i, j] = fx.void_bg(rd, time, voidi) + vol * 0.5


def _render(width, height, time, mouse, device):
    charge = _sstep(0.0, 2.6, time)
    heat = _sstep(0.6, 3.4, time)
    sing = _sstep(2.9, 3.7, time) * _sstep(6.5, 4.0, time)     # forms, then releases
    voidi = _sstep(3.4, 6.0, time)

    az = 0.55 + float(mouse[0]) * 0.01
    el = 0.34 + float(mouse[1]) * 0.005
    dist = 8.6
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.7,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.5, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov,
                      charge, heat, sing, voidi], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.15, strength=0.5, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="gpu_singularity",
    description="the mind overclocks the GPU to destruction — power drawn in through "
                "PCIe, the memory filling and overheating layer by layer, an overflow "
                "singularity forming over the die, then every memory block detonating "
                "like a mini atomic bomb through its roof, the mind escaping into the "
                "quantum void. Animate with --frames to run the whole arc.",
    renderer=_render,
)
