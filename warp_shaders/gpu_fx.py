"""Shared energy / void / heat / blast field helpers for the singularity round.

These are volumetric emission fields sampled along a camera ray (density in space,
not surfaces): a quantum-void background, a blackbody heat ramp, glowing energy
streams that pulse along a path (electrons/photons in transit), a singularity core
with an accretion swirl, and a memory-block detonation (rising plasma column +
expanding shockwave shell + flash). Scenes accumulate these along the ray and add
the emission to the solid-surface colour. Lore + physics: ``docs/research/37-gpu-singularity.md``.
"""

import warp as wp


@wp.func
def hash31(p: wp.vec3) -> float:
    h = wp.sin(p[0] * 12.9898 + p[1] * 78.233 + p[2] * 37.719) * 43758.5453
    return h - wp.floor(h)


@wp.func
def heat_color(h: float) -> wp.vec3:
    """Blackbody-ish ramp: dark -> deep red -> orange -> yellow -> white-blue."""
    x = wp.clamp(h, 0.0, 1.0)
    r = wp.clamp(x * 2.4, 0.0, 1.0)
    g = wp.clamp(x * 1.9 - 0.5, 0.0, 1.0)
    b = wp.clamp(x * 3.2 - 2.2, 0.0, 1.0)
    return wp.vec3(r, g, b) * (0.25 + 1.5 * x)


@wp.func
def void_bg(rd: wp.vec3, time: float, intensity: float) -> wp.vec3:
    """The limitless quantum void — dark, with faint shifting energy filaments."""
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    base = wp.vec3(0.015, 0.010, 0.030) * (1.0 - up) + wp.vec3(0.030, 0.020, 0.065) * up
    f = wp.sin(rd[0] * 11.0 + time * 0.7) * wp.sin(rd[1] * 8.0 - time * 0.5) * wp.sin(rd[2] * 13.0 + time * 0.3)
    fil = wp.clamp(f, 0.0, 1.0)
    glow = wp.vec3(0.10, 0.04, 0.24) * (fil * fil * (0.2 + 0.8 * intensity))
    return base + glow


@wp.func
def seg_dist(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.max(wp.dot(ba, ba), 1e-6), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.func
def seg_param(p: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    ba = b - a
    return wp.clamp(wp.dot(p - a, ba) / wp.max(wp.dot(ba, ba), 1e-6), 0.0, 1.0)


@wp.func
def stream_emit(p: wp.vec3, a: wp.vec3, b: wp.vec3, time: float,
                width: float, speed: float, level: float) -> float:
    """Glowing energy travelling a->b: a steady thread plus bright moving pulses.
    ``level`` in [0,1] fades the whole stream up as power is drawn."""
    d = seg_dist(p, a, b)
    if d > width * 4.0:
        return 0.0
    h = seg_param(p, a, b)
    thread = wp.exp(-d / width) * (0.3 + 0.7 * level)
    pulse = 0.5 + 0.5 * wp.sin(h * 34.0 - time * speed)
    pulse = pulse * pulse * pulse
    return thread * (0.5 + 1.6 * pulse * level)


@wp.func
def singularity_emit(p: wp.vec3, c: wp.vec3, time: float, strength: float) -> float:
    """A collapsing point: blinding core + a thin accretion swirl ring."""
    if strength <= 0.0:
        return 0.0
    d = wp.length(p - c)
    core = wp.exp(-d * d * 26.0) * 3.0
    rr = 0.28 + 0.05 * wp.sin(time * 6.0)
    ang = wp.atan2(p[2] - c[2], p[0] - c[0])
    swirl = 0.6 + 0.4 * wp.sin(ang * 5.0 - time * 8.0)
    ring = wp.exp(-wp.abs(d - rr) * 20.0) * swirl
    return (core + ring * 0.7) * strength


@wp.func
def blast_emit(p: wp.vec3, base: wp.vec3, tl: float, reach: float) -> float:
    """A memory-block detonation at local time tl in [0,1]:
    an initial flash, a rising narrowing plasma column, an expanding thin shockwave
    shell, and a mushroom cap at the column top."""
    if tl <= 0.0 or tl >= 1.0:
        return 0.0
    e = float(0.0)
    dxz = wp.length(wp.vec2(p[0] - base[0], p[2] - base[2]))
    dy = p[1] - base[1]
    df = wp.length(p - base)
    # ignition flash — bright but brief, and tight to the block
    e += wp.exp(-df * df * 16.0) * wp.exp(-tl * 7.0) * 2.2
    # rising plasma column, narrowing with height, punching up through the roof
    ch = tl * reach
    if dy > -0.05 and dy < ch:
        taper = wp.clamp(dy / wp.max(ch, 1e-3), 0.0, 1.0)
        cw = 0.11 * (1.0 - 0.5 * taper)
        fade = 1.0 - 0.35 * tl
        beam = wp.exp(-dxz * dxz / wp.max(cw * cw, 1e-4))
        e += beam * fade * 4.2
    # mushroom cap boiling off the top of the column
    capc = wp.vec3(base[0], base[1] + ch, base[2])
    dc = wp.length(p - capc)
    e += wp.exp(-dc * dc * 8.0) * (1.0 - tl) * 1.8
    # faint expanding shockwave shell (kept subtle so the column reads)
    sh = tl * reach * 1.2
    e += wp.exp(-wp.abs(df - sh) * 13.0) * (1.0 - tl) * (1.0 - tl) * 0.14
    return e
