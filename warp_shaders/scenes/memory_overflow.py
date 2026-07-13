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
from ..blast.render import _cloud_density, _fireball_density
from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..scene import Scene

_MAXD = 30.0
_BLK = wp.constant(wp.vec3(0.72, 0.46, 0.56))   # block half-extents
_TOP = 0.46
_BURST = wp.constant(wp.vec3(0.0, 0.5, 0.0))    # detonation ground zero on the block


@wp.func
def _mushroom(p: wp.vec3, bt: float, r_fb: float, cap_y: float, cap_r: float,
              stem_r: float, core_k: float) -> wp.vec4:
    """A proper (small) nuclear mushroom off the block top — the engine blast model."""
    fb_glow = kelvin_to_rgb(core_k) * 0.5
    col = wp.vec3(0.0, 0.0, 0.0)
    dens = float(0.0)
    fd = _fireball_density(p, _BURST, r_fb, bt)
    if fd > 0.001:
        rn = wp.length(p - _BURST) / (r_fb + 1e-3)
        temp = core_k * (1.0 - 0.45 * wp.clamp(rn, 0.0, 1.0))
        bright = 0.3 + 1.6 * wp.smoothstep(1.0, 0.0, rn)
        col += kelvin_to_rgb(temp) * (fd * 9.0 * bright)
        dens = wp.max(dens, wp.clamp(fd * 2.4, 0.0, 1.0))
    cd = _cloud_density(p, _BURST, cap_y, cap_r, stem_r, r_fb * 0.4, bt)
    rr = wp.length(wp.vec2(p[0] - _BURST[0], p[2] - _BURST[2]))
    yy = p[1] - _BURST[1]
    if yy > -0.1 and yy < (cap_y - _BURST[1]) + 0.2:
        cw = stem_r * (1.0 + 0.5 * wp.clamp(yy / (cap_y + 0.3), 0.0, 1.0))
        stemd = wp.exp(-(rr * rr) / (cw * cw + 1e-4)) * wp.smoothstep(cap_y + 0.3, _BURST[1] - 0.15, p[1])
        cd = wp.max(cd, stemd * 0.9)
    if cd > 0.001:
        under = wp.exp(-wp.length(p - _BURST) / (2.5 * r_fb + 1e-3))
        hn = wp.clamp(p[1] / (cap_y + cap_r + 1e-3), 0.0, 1.0)
        dust = wp.vec3(0.24, 0.17, 0.11)
        crown = wp.vec3(0.66, 0.67, 0.70)
        body = dust + (crown - dust) * wp.smoothstep(0.32, 0.78, hn)
        topl = wp.clamp((p[1] - _BURST[1]) / (cap_r * 1.6) + 0.5, 0.0, 1.0)
        body = body * (0.32 + 0.68 * topl)
        col += (body + fb_glow * (under * 2.4)) * cd
        dens = wp.max(dens, wp.clamp(cd * 1.6, 0.0, 1.0))
    return wp.vec4(col[0], col[1], col[2], dens)


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
                   sing: float, bl: float, voidi: float, bt: float, r_fb: float,
                   cap_y: float, cap_r: float, stem_r: float, core_k: float):
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

    if hit == 0:
        surf = fx.void_bg(rd, time, voidi)

    # volumetrics: singularity pinch above the block (additive glow) + a proper
    # mushroom cloud (absorptive smoke, occludes the block)
    steps = 60
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    trans = float(1.0)
    smoke = wp.vec3(0.0, 0.0, 0.0)
    glow = wp.vec3(0.0, 0.0, 0.0)
    sc = wp.vec3(0.0, _TOP + 0.25 + sing * 0.5, 0.0)
    for _ in range(steps):
        pv = eye + rd * tv
        se = fx.singularity_emit(pv, sc, time, sing)
        glow += wp.vec3(0.75, 0.5, 1.0) * (se * dt)
        if bt > 0.0 and trans > 0.02:
            m = _mushroom(pv, bt, r_fb, cap_y, cap_r, stem_r, core_k)
            a = wp.clamp(m[3] * 1.9 * dt, 0.0, 1.0)
            smoke += wp.vec3(m[0], m[1], m[2]) * (a * trans)
            trans = trans * (1.0 - a)
        tv += dt

    img[i, j] = surf * trans + smoke + glow


def _render(width, height, time, mouse, device):
    fill = _sstep(0.0, 2.6, time)
    heat = _sstep(0.8, 3.2, time)
    sing = _sstep(2.9, 3.5, time) * _sstep(4.4, 3.7, time)
    bl = (time - 3.6) / 1.4
    voidi = _sstep(3.2, 5.5, time)

    # the (small) mushroom, growing from t=3.7s off the block top
    bt = time - 3.7
    r_fb = wp.clamp(bt * 1.4, 0.0, 1.0) * 0.38
    rise = wp.max(bt, 0.0)
    cap_y = _BURST[1] + wp.clamp(rise, 0.0, 2.4) * 0.62
    cap_r = r_fb * (1.4 + 0.8 * wp.min(rise / 3.0, 1.0)) + 0.1 * wp.clamp(rise, 0.0, 2.5)
    stem_r = r_fb * 0.95 + 0.06
    core_k = 7000.0

    pull = _sstep(3.6, 5.5, time)
    az = 0.6 + float(mouse[0]) * 0.01
    el = 0.24 + 0.04 * pull + float(mouse[1]) * 0.005
    dist = 4.6 + 1.8 * pull
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.5 + 0.5 * pull,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.35 + 0.85 * pull, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(48.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov,
                      fill, heat, sing, bl, voidi, bt, r_fb, cap_y, cap_r,
                      stem_r, core_k], device=device)
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
