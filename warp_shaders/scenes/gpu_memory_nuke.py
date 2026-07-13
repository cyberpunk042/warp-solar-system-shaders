"""GPU memory nuke — the GDDR blocks around the die detonate one by one.

A companion to ``gpu_singularity``. Same real RTX 6000 Pro Blackwell board, same
electrons drawn through it — but this time the **die survives** and the overload is
pushed out into the **memory**. Each GDDR7 package around the GPU overflows and goes
off in its own **mushroom cloud**, one after another, a rolling chain of little
atomic bombs sweeping across the board while the die glows white-hot at the centre,
feeding them. Thirteen packages, thirteen mushrooms. Animate over ``--frames`` to run
the whole chain. See ``docs/research/37-gpu-singularity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import gpu_fx as fx
from ..blast.render import _cloud_density, _fireball_density
from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..scene import Scene
from .gpu_board import _die_top, _mem, board_map, board_shade
from .gpu_singularity import _bao, _bnormal, _flow, _gddr, _sstep

_MAXD = 60.0
_DIE = wp.constant(wp.vec3(-0.75, 0.30, 0.05))
_CORE_K = 6800.0


@wp.func
def _block_start(k: int) -> float:
    # detonation order sweeps left -> right across the board by x position, so the
    # top-row and bottom-row packages that share an x pop together as a wave passes
    gx = _gddr(k)[0]
    return 3.0 + (gx + 2.5) * 1.35


@wp.func
def _mushroom_at(p: wp.vec3, base: wp.vec3, bt: float, r_fb: float, cap_y: float,
                 cap_r: float, stem_r: float, fade: float) -> wp.vec4:
    """A proper (small) nuclear mushroom off `base` — the engine blast model, gated
    by a cheap bounding test so the expensive turbulence only runs inside the cloud."""
    if bt <= 0.0 or fade <= 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    rr = wp.length(wp.vec2(p[0] - base[0], p[2] - base[2]))
    yy = p[1] - base[1]
    boundr = wp.max(cap_r * 1.7, stem_r * 2.5) + 0.2
    if rr > boundr or yy < -0.25 or yy > (cap_y - base[1]) + cap_r + 0.35:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)

    fb_glow = kelvin_to_rgb(_CORE_K) * 0.5
    col = wp.vec3(0.0, 0.0, 0.0)
    dens = float(0.0)
    fd = _fireball_density(p, base, r_fb, bt)
    if fd > 0.001:
        rn = wp.length(p - base) / (r_fb + 1e-3)
        temp = _CORE_K * (1.0 - 0.45 * wp.clamp(rn, 0.0, 1.0))
        bright = 0.3 + 1.6 * wp.smoothstep(1.0, 0.0, rn)
        col += kelvin_to_rgb(temp) * (fd * 9.0 * bright)
        dens = wp.max(dens, wp.clamp(fd * 2.4, 0.0, 1.0))
    cd = _cloud_density(p, base, cap_y, cap_r, stem_r, r_fb * 0.4, bt)
    if yy > -0.1 and yy < (cap_y - base[1]) + 0.25:
        cw = stem_r * (1.0 + 0.5 * wp.clamp(yy / (cap_y + 0.3), 0.0, 1.0))
        stemd = wp.exp(-(rr * rr) / (cw * cw + 1e-4)) * wp.smoothstep(cap_y + 0.3, base[1] - 0.15, p[1])
        cd = wp.max(cd, stemd * 0.9)
    if cd > 0.001:
        under = wp.exp(-wp.length(p - base) / (2.5 * r_fb + 1e-3))
        hn = wp.clamp((p[1] - base[1] + 0.2) / (cap_y - base[1] + cap_r + 1e-3), 0.0, 1.0)
        dust = wp.vec3(0.24, 0.17, 0.11)
        crown = wp.vec3(0.66, 0.67, 0.70)
        body = dust + (crown - dust) * wp.smoothstep(0.32, 0.78, hn)
        topl = wp.clamp((p[1] - base[1]) / (cap_r * 1.6) + 0.5, 0.0, 1.0)
        body = body * (0.32 + 0.68 * topl)
        col += (body + fb_glow * (under * 2.4)) * cd
        dens = wp.max(dens, wp.clamp(cd * 1.6, 0.0, 1.0))
    return wp.vec4(col[0] * fade, col[1] * fade, col[2] * fade, dens * fade)


@wp.func
def _block_params(k: int, time: float) -> wp.vec4:
    """(bt, r_fb, cap_y, cap_r) packed; stem_r + fade derived by the caller."""
    a = time - _block_start(k)
    r_fb = wp.clamp(a * 1.5, 0.0, 1.0) * 0.36
    rise = wp.max(a, 0.0)
    cap_y = _gddr(k)[1] + 0.14 + wp.clamp(rise, 0.0, 3.5) * 0.62
    cap_r = r_fb * (1.5 + 0.9 * wp.min(rise / 3.0, 1.0)) + 0.13 * wp.clamp(rise, 0.0, 3.5)
    return wp.vec4(a, r_fb, cap_y, cap_r)


@wp.func
def _mem_field(p: wp.vec3, time: float) -> wp.vec4:
    """Combined emission + density of all active memory-block mushrooms at p."""
    col = wp.vec3(0.0, 0.0, 0.0)
    dens = float(0.0)
    for k in range(13):
        pr = _block_params(k, time)
        a = pr[0]
        if a > 0.0 and a < 8.0:
            fade = wp.clamp(1.5 - a / 6.5, 0.0, 1.0)          # dissipate slowly so a wave stays up
            stem_r = pr[1] * 0.95 + 0.06
            m = _mushroom_at(p, _gddr(k), a, pr[1], pr[2], pr[3], stem_r, fade)
            col += wp.vec3(m[0], m[1], m[2])
            dens = wp.max(dens, m[3])
    return wp.vec4(col[0], col[1], col[2], dens)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, charge: float, heat: float,
                   voidi: float, efade: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.05)
    hit = int(0)
    for _ in range(220):
        p = eye + rd * t
        d = board_map(p)
        if d < 0.0008 * t + 0.0005:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    t_hit = _MAXD
    surf = fx.void_bg(rd, time, voidi)
    if hit == 1:
        t_hit = t
        p = eye + rd * t
        n = _bnormal(p)
        ao = _bao(p, n)
        base = board_shade(p, n, rd, ao, 0.0)
        if _mem(p) < 0.02:                                    # a GDDR block
            kbest = int(0)
            db = float(1e9)
            for k in range(13):
                dk = wp.length(p - _gddr(k))
                if dk < db:
                    db = dk
                    kbest = k
            base = base * 0.6 + fx.heat_color(heat) * (0.25 + 0.85 * heat)
            a = time - _block_start(kbest)
            if a > 0.0:
                base = base * wp.clamp(1.0 - a * 2.2, 0.0, 1.0)   # block blown open
        elif _die_top(p) < 0.02 and p[1] > 0.18:              # the die survives, glows hot
            base = base * 0.75 + fx.heat_color(heat * 0.8) * (0.25 + 0.55 * heat)
        surf = base

    # volumetric pass front-to-back: electrons (additive) + the chain of mushrooms
    t_end = t_hit
    if hit == 0:
        t_end = 42.0
    steps = 58
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    trans = float(1.0)
    smoke = wp.vec3(0.0, 0.0, 0.0)
    glow = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(steps):
        pv = eye + rd * tv
        glow += _flow(pv, time, charge) * (efade * dt)
        if trans > 0.02:
            m = _mem_field(pv, time)
            a = wp.clamp(m[3] * 1.9 * dt, 0.0, 1.0)
            smoke += wp.vec3(m[0], m[1], m[2]) * (a * trans)
            trans = trans * (1.0 - a)
        tv += dt

    img[i, j] = surf * trans + smoke + glow


def _render(width, height, time, mouse, device):
    charge = _sstep(0.0, 3.0, time)
    heat = _sstep(1.4, 3.4, time)
    voidi = _sstep(3.0, 6.0, time)
    efade = 1.0 - _sstep(3.0, 4.0, time)      # electrons consumed as the chain starts

    pull = _sstep(2.6, 5.5, time)
    az = 0.66 + time * 0.02 + float(mouse[0]) * 0.01
    el = 0.32 - 0.02 * pull + float(mouse[1]) * 0.005
    dist = 8.4 + 2.6 * pull
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.7 + 0.4 * pull,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(-0.4, 0.2 + 0.6 * pull, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(52.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov,
                      charge, heat, voidi, efade], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.45, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="gpu_memory_nuke",
    description="the RTX board with the GDDR memory going off instead of the die — each of "
                "the thirteen memory packages around the GPU overflows and detonates in its "
                "own mushroom cloud, one by one in a rolling chain across the board, while "
                "the die survives white-hot at the centre feeding them. Animate with --frames.",
    renderer=_render,
)
