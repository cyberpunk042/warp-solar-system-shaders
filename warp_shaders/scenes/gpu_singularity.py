"""GPU singularity — the mind overclocks an RTX 6000 Pro Blackwell until it goes off.

The real board from ``gpu_board`` (the dense workstation card — huge exposed die, a
full GDDR7 ring, the multi-phase VRM, the 12VHPWR connector, PCIe fingers) is the
stage, laid flat so the blast rises off it. The mind inside the die revs up and
**draws power through the actual board**: electrons stream in as cold blue current
from the 12VHPWR connector through the VRM chokes into the die, and up the PCIe edge,
then the die pushes charge out to the GDDR ring — real current over the real copper.
The memory fills and overheats red→white, an overflow **singularity** pinches above the
die, each GDDR block pops like a mini atomic bomb — and then the die itself lets go in a
**proper mushroom cloud** (the engine's nuclear-fireball model): incandescent fireball,
rising stem, billowing condensation cap. The mind escapes into the quantum void. Animate
over ``--frames`` to run the whole arc. See ``docs/research/37-gpu-singularity.md``.
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

_MAXD = 60.0
_DIE = wp.constant(wp.vec3(-0.75, 0.30, 0.05))          # die centre (world = board-local)
_PWR = wp.constant(wp.vec3(3.15, 0.34, 0.95))           # 12VHPWR connector
_VRM = wp.constant(wp.vec3(1.6, 0.30, 0.9))             # VRM choke bank midpoint
_BURST = wp.constant(wp.vec3(-0.75, 0.40, 0.05))        # nuke ground zero, on the die


@wp.func
def _sstep(a: float, b: float, x: float) -> float:
    t = wp.clamp((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def _bnormal(p: wp.vec3) -> wp.vec3:
    e = 0.0011
    dx = board_map(p + wp.vec3(e, 0.0, 0.0)) - board_map(p - wp.vec3(e, 0.0, 0.0))
    dy = board_map(p + wp.vec3(0.0, e, 0.0)) - board_map(p - wp.vec3(0.0, e, 0.0))
    dz = board_map(p + wp.vec3(0.0, 0.0, e)) - board_map(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _bao(p: wp.vec3, n: wp.vec3) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = board_map(p + n * hr)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.func
def _gddr(k: int) -> wp.vec3:
    """Centre of GDDR7 package k (0..12), matching gpu_board._mem's three groups:
    a top row (z=1.2), a bottom row (z=-1.1), and a left column (x=-2.35)."""
    if k < 5:
        return wp.vec3(-1.9 + 0.56 * float(k), 0.16, 1.2)
    if k < 10:
        return wp.vec3(-1.9 + 0.56 * float(k - 5), 0.16, -1.1)
    return wp.vec3(-2.35, 0.16, -0.55 + 0.55 * float(k - 10))


@wp.func
def _pop_local(k: int, time: float) -> float:
    # each memory block pops in turn, from t=4.5s, ~0.09s apart, ~0.8s each
    return (time - (4.5 + float(k) * 0.09)) / 0.8


@wp.func
def _flow(p: wp.vec3, time: float, charge: float) -> wp.vec3:
    """Electrons + photons in transit over the *real* board: 12VHPWR -> VRM -> die,
    PCIe edge -> die, then die -> each GDDR block. Cold-blue electron current with
    white photon flashes riding along."""
    col = wp.vec3(0.0, 0.0, 0.0)
    spd = 7.0 + 11.0 * charge
    # power in: 12VHPWR -> VRM chokes -> die
    e0 = fx.stream_emit(p, _PWR, _VRM, time, 0.07, spd, charge)
    col += wp.vec3(0.3, 0.65, 1.15) * (e0 * 3.2)
    e1 = fx.stream_emit(p, _VRM, _DIE, time, 0.07, spd, charge)
    col += wp.vec3(0.3, 0.65, 1.15) * (e1 * 3.2)
    ph = fx.stream_emit(p, _PWR, _DIE, time * 1.4 + 3.0, 0.04, spd * 1.3, charge)
    col += wp.vec3(1.0, 0.95, 0.75) * (ph * 2.2)             # photon flashes
    # PCIe edge -> die (two lanes)
    for k in range(2):
        a = wp.vec3(-0.4 + float(k) * 1.0, 0.06, -1.45)
        pe = fx.stream_emit(p, a, _DIE, time, 0.06, spd, charge)
        col += wp.vec3(0.35, 0.72, 1.15) * (pe * 2.6)
    # die -> each GDDR block: charge pushed out to fill the memory
    for k in range(13):
        sm = fx.stream_emit(p, _DIE, _gddr(k), time, 0.06, spd * 0.85, charge)
        col += wp.vec3(0.4, 0.82, 1.05) * (sm * 2.2)
    return col


@wp.func
def _pops(p: wp.vec3, time: float) -> wp.vec3:
    """The staggered mini-detonations of the individual GDDR blocks."""
    col = wp.vec3(0.0, 0.0, 0.0)
    for k in range(13):
        tl = _pop_local(k, time)
        b = fx.blast_emit(p, _gddr(k), tl, 1.4)
        hot = wp.clamp(1.1 - tl, 0.0, 1.0)
        col += fx.heat_color(0.45 + 0.55 * hot) * b
    return col


@wp.func
def _mushroom(p: wp.vec3, time: float, bt: float, r_fb: float, cap_y: float,
              cap_r: float, stem_r: float, core_k: float) -> wp.vec4:
    """The proper nuclear mushroom off the die (engine blast model). Returns
    (rgb_emission, density) — the caller composites it front-to-back with
    transmittance so the smoke occludes the board behind it."""
    fb_glow = kelvin_to_rgb(core_k) * 0.5
    col = wp.vec3(0.0, 0.0, 0.0)
    dens = float(0.0)
    # incandescent fireball (blackbody), brightest at the core
    fd = _fireball_density(p, _BURST, r_fb, bt)
    if fd > 0.001:
        rn = wp.length(p - _BURST) / (r_fb + 1e-3)
        temp = core_k * (1.0 - 0.45 * wp.clamp(rn, 0.0, 1.0))
        bright = 0.3 + 1.6 * wp.smoothstep(1.0, 0.0, rn)
        col += kelvin_to_rgb(temp) * (fd * 9.0 * bright)
        dens = wp.max(dens, wp.clamp(fd * 2.4, 0.0, 1.0))
    # dust-laden brown stem -> white condensation crown, dark underside, orange underlight.
    # Pass a small carve radius (r_fb*0.4) so the cloud is NOT hollowed all the way up the
    # column — the stem stays connected to the fireball instead of floating free.
    cd = _cloud_density(p, _BURST, cap_y, cap_r, stem_r, r_fb * 0.4, bt)
    # explicit rising column so the mushroom always reads as one connected shape
    rr = wp.length(wp.vec2(p[0] - _BURST[0], p[2] - _BURST[2]))
    yy = p[1] - _BURST[1]
    if yy > -0.1 and yy < (cap_y - _BURST[1]) + 0.3:
        cw = stem_r * (1.0 + 0.5 * wp.clamp(yy / (cap_y + 0.5), 0.0, 1.0))   # flares upward
        stemd = wp.exp(-(rr * rr) / (cw * cw + 1e-4)) * wp.smoothstep(cap_y + 0.4, _BURST[1] - 0.2, p[1])
        cd = wp.max(cd, stemd * 0.9)
    if cd > 0.001:
        under = wp.exp(-wp.length(p - _BURST) / (2.5 * r_fb + 1e-3))
        hn = wp.clamp(p[1] / (cap_y + cap_r + 1e-3), 0.0, 1.0)
        dust = wp.vec3(0.24, 0.17, 0.11)
        crown = wp.vec3(0.66, 0.67, 0.70)
        body = dust + (crown - dust) * wp.smoothstep(0.32, 0.78, hn)
        topl = wp.clamp((p[1] - _BURST[1]) / (cap_r * 1.6) + 0.5, 0.0, 1.0)
        body = body * (0.32 + 0.68 * topl)
        smoke = body + fb_glow * (under * 2.4)
        col += smoke * cd
        dens = wp.max(dens, wp.clamp(cd * 1.6, 0.0, 1.0))
    return wp.vec4(col[0], col[1], col[2], dens)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, charge: float, heat: float,
                   sing: float, voidi: float, bt: float, r_fb: float, cap_y: float,
                   cap_r: float, stem_r: float, core_k: float, efade: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    # solid board march (the real RTX board, axis-aligned in world)
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
        dm = _mem(p)
        dd = _die_top(p)
        if dm < 0.02:                                        # a GDDR block
            kbest = int(0)
            db = float(1e9)
            for k in range(13):
                dk = wp.length(p - _gddr(k))
                if dk < db:
                    db = dk
                    kbest = k
            fillh = wp.clamp((p[1] - 0.07) / 0.16, 0.0, 1.0)  # charge fills bottom-up
            mh = wp.clamp(heat * 1.3 - fillh * 0.2, 0.0, 1.0)
            base = base * 0.6 + fx.heat_color(mh) * (0.25 + 0.85 * heat)
            tl = _pop_local(kbest, time)
            if tl > 0.0:
                base = base * wp.clamp(1.0 - tl * 2.2, 0.0, 1.0)   # block blown open
        elif dd < 0.02 and p[1] > 0.18:                      # the die top
            base = base * 0.7 + fx.heat_color(heat) * (0.3 + 0.7 * heat)
        # scorch the whole board as the fireball blooms
        base = base * wp.clamp(1.0 - _sstep(0.0, 1.0, bt) * 0.7, 0.15, 1.0)
        surf = base

    # volumetric pass front-to-back: electrons + singularity + block pops (additive
    # glow) and the big mushroom (absorptive smoke that occludes the board)
    t_end = t_hit
    if hit == 0:
        t_end = 42.0
    steps = 56
    dt = (t_end - 0.05) / float(steps)
    tv = float(0.06)
    trans = float(1.0)
    smoke = wp.vec3(0.0, 0.0, 0.0)
    glow = wp.vec3(0.0, 0.0, 0.0)
    sc = wp.vec3(_DIE[0], 0.55 + sing * 0.7, _DIE[2])
    for _ in range(steps):
        pv = eye + rd * tv
        glow += _flow(pv, time, charge) * (efade * dt)
        se = fx.singularity_emit(pv, sc, time, sing)
        glow += wp.vec3(0.8, 0.55, 1.0) * (se * dt)
        glow += _pops(pv, time) * dt
        if bt > 0.0 and trans > 0.02:
            m = _mushroom(pv, time, bt, r_fb, cap_y, cap_r, stem_r, core_k)
            a = wp.clamp(m[3] * 1.8 * dt, 0.0, 1.0)
            smoke += wp.vec3(m[0], m[1], m[2]) * (a * trans)
            trans = trans * (1.0 - a)
        tv += dt

    img[i, j] = surf * trans + smoke + glow


def _render(width, height, time, mouse, device):
    charge = _sstep(0.0, 3.0, time)
    heat = _sstep(1.6, 4.4, time)
    sing = _sstep(3.6, 4.5, time) * _sstep(5.6, 4.6, time)     # forms, then the burst eats it
    voidi = _sstep(4.5, 7.0, time)
    efade = 1.0 - _sstep(4.8, 5.6, time)                       # electrons consumed at the burst

    # the nuclear mushroom, growing from t=5.0s
    bt = time - 5.0
    r_fb = wp.clamp(bt * 1.3, 0.0, 1.0) * 0.95                 # fireball radius
    rise = wp.max(bt, 0.0)
    lift = wp.clamp(rise, 0.0, 5.0) * 0.95
    cap_y = _BURST[1] + lift                                   # cap rides the rising column
    cap_r = r_fb * (1.4 + 0.8 * wp.min(rise / 5.0, 1.0)) + 0.2 * wp.clamp(rise, 0.0, 4.0)
    stem_r = r_fb * 0.95 + 0.12
    core_k = 7200.0

    # camera dollies back + tilts up as the cloud grows: close on the board while the
    # electrons flow, then pull out to frame the whole mushroom
    pull = _sstep(4.4, 7.5, time)
    az = 0.62 + time * 0.03 + float(mouse[0]) * 0.01
    el = 0.34 - 0.05 * pull + float(mouse[1]) * 0.005
    dist = 7.6 + 5.4 * pull
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el) + 0.7 + 0.9 * pull,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(-0.5, 0.2 + 1.7 * pull, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(50.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov,
                      charge, heat, sing, voidi, bt, r_fb, cap_y, cap_r, stem_r,
                      core_k, efade], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.45, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="gpu_singularity",
    description="an RTX 6000 Pro Blackwell board overclocked to destruction — electrons "
                "drawn in through the real 12VHPWR/VRM/PCIe path into the die, the GDDR "
                "memory overheating and each block popping, an overflow singularity, then "
                "the die going off in a proper nuclear mushroom cloud. Animate with --frames.",
    renderer=_render,
)
