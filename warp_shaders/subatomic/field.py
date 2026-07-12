"""Sub-atomic field primitives — the physics behind the particle scenes.

Point-emission ``@wp.func`` building blocks that a scene kernel integrates along
its ray (additive HDR emission, no absorption): a colour-charged **quark plasma**,
a QCD **gluon flux tube** (a textured, flowing colour string), a confinement
**bag** glow, and the analytic hydrogen **orbital density** |ψ_{nlm}|². Grounded
in ``docs/research/21-standard-model.md``.

Everything here is stylised but structurally faithful — correct quark content,
colour charge summing to neutral, real orbital shapes (nodes + lobes), flux-tube
confinement. Not to physical scale (a nucleus is ~1e-5 of an atom).
"""

import warp as wp

from ..procedural.noise import fbm3, ridged3, value3


# ---------------------------------------------------------------- geometry
@wp.func
def sd_capsule(p: wp.vec3, a: wp.vec3, b: wp.vec3, r: float) -> float:
    """Signed distance to a capsule (line segment a→b, radius r)."""
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h) - r


@wp.func
def void(rd: wp.vec3) -> wp.vec3:
    """A near-black void with a faint dusting of stars (so a particle reads as
    isolated in space, not floating on flat black)."""
    s = value3(rd * 190.0)
    star = wp.pow(s, 60.0) * 0.7
    base = wp.vec3(0.006, 0.008, 0.014) * (0.5 + 0.5 * rd[1])
    return base + wp.vec3(0.8, 0.85, 1.0) * star


# ---------------------------------------------------------------- quark plasma
@wp.func
def quark_emit(p: wp.vec3, center: wp.vec3, radius: float, col: wp.vec3,
               time: float, seed: float) -> wp.vec3:
    """Emission at point ``p`` from a colour-charged quark: a hot white core
    inside a turbulent, colour-charged plasma shell (fBm-modulated), falling off
    past ``radius``. Returns an additive colour contribution."""
    d = wp.length(p - center)
    x = d / radius
    if x > 1.6:
        return wp.vec3(0.0, 0.0, 0.0)
    # turbulent plasma texture in the quark's own frame
    q = (p - center) * (3.2 / radius)
    turb = fbm3(q + wp.vec3(seed, seed * 1.7, time * 0.9), 4)
    fil = ridged3(q * 1.7 + wp.vec3(0.0, time * 0.6, seed), 3)
    core = wp.exp(-x * x * 9.0)                      # bright core
    shell = wp.exp(-x * x * 4.2) * (0.3 + 0.8 * turb) * (0.5 + 0.7 * fil)
    white = wp.vec3(1.0, 0.96, 0.9)
    return white * (core * 1.2) + col * (shell * 1.1)


# ---------------------------------------------------------------- gluon flux tube
@wp.func
def tube_emit(p: wp.vec3, a: wp.vec3, b: wp.vec3, radius: float, col: wp.vec3,
              time: float) -> wp.vec3:
    """Emission at ``p`` from a QCD gluon **flux tube** binding two quarks: a taut
    capsule whose surface flows with 1-D noise along its length and pulses in
    time — the colour string of confinement."""
    ba = b - a
    L = wp.length(ba)
    if L < 1.0e-5:
        return wp.vec3(0.0, 0.0, 0.0)
    dir = ba / L
    pa = p - a
    h = wp.clamp(wp.dot(pa, dir) / L, 0.0, 1.0)      # 0..1 along the tube
    axis = a + ba * h
    r = wp.length(p - axis)
    if r > radius * 2.2:
        return wp.vec3(0.0, 0.0, 0.0)
    # gentle flowing texture along the tube + radial confinement profile
    flow = 0.75 + 0.25 * wp.sin(h * 7.0 - time * 5.0)
    ripple = 0.85 + 0.25 * value3(wp.vec3(h * 9.0, time * 1.2, 0.0))
    prof = wp.exp(-(r / radius) * (r / radius) * 3.0)
    # ends are anchored on the quarks (brightest), middle is the taut string
    taut = 0.45 + 0.55 * (1.0 - 4.0 * (h - 0.5) * (h - 0.5))
    e = prof * flow * ripple * taut
    return col * (e * 0.9) + wp.vec3(0.7, 0.85, 1.0) * (prof * flow * 0.6)


# ---------------------------------------------------------------- confinement bag
@wp.func
def bag_glow(p: wp.vec3, center: wp.vec3, radius: float, tint: wp.vec3) -> wp.vec3:
    """Faint volumetric glow of the confinement 'bag' (MIT bag model boundary)
    that holds the quarks — brightest as a soft shell near ``radius``."""
    d = wp.length(p - center) / radius
    shell = wp.exp(-(d - 0.9) * (d - 0.9) * 10.0)
    return tint * (shell * 0.05)


# ---------------------------------------------------------------- hydrogen orbitals
@wp.func
def orbital_psi2(p: wp.vec3, orb: int, a0: float) -> float:
    """Unnormalised hydrogen probability density |ψ_{nlm}(p)|² for a few orbitals,
    ``a0`` = Bohr radius in world units. The y-axis is the quantisation axis.

    orb: 0=1s, 1=2s, 2=2p_z, 3=3p_z, 4=3d_z², 5=3d_(x²−y²) cloverleaf.
    Real radial (Laguerre) × angular (spherical-harmonic) shapes — nodes + lobes.
    """
    r = wp.length(p) + 1.0e-6
    ct = p[1] / r                                    # cos(theta), y = axis
    rho = r / a0
    if orb == 0:                                     # 1s : e^{-2ρ}
        f = wp.exp(-2.0 * rho)
        return f
    if orb == 1:                                     # 2s : (2−ρ)² e^{−ρ}
        g = 2.0 - rho
        return g * g * wp.exp(-rho) * 0.25
    if orb == 2:                                     # 2p_z : ρ² e^{−ρ} cos²θ
        return rho * rho * wp.exp(-rho) * ct * ct * 0.15
    if orb == 3:                                     # 3p_z : ρ²(4−ρ)² e^{−2ρ/3} cos²θ
        g = 4.0 - rho
        return rho * rho * g * g * wp.exp(-0.6667 * rho) * ct * ct * 0.004
    if orb == 4:                                     # 3d_z² : ρ⁴ e^{−2ρ/3} (3cos²θ−1)²
        a = 3.0 * ct * ct - 1.0
        return rho * rho * rho * rho * wp.exp(-0.6667 * rho) * a * a * 0.0016
    # orb == 5 : 3d cloverleaf : ρ⁴ e^{−2ρ/3} sin⁴θ cos²(2φ)
    st2 = wp.max(1.0 - ct * ct, 0.0)
    phi = wp.atan2(p[2], p[0])
    c2 = wp.cos(2.0 * phi)
    return rho * rho * rho * rho * wp.exp(-0.6667 * rho) * st2 * st2 * c2 * c2 * 0.0016


@wp.func
def flavor_color(flav: int) -> wp.vec3:
    """A per-flavour tint for the six quarks (up,down,charm,strange,top,bottom)."""
    if flav == 0:
        return wp.vec3(1.0, 0.42, 0.36)              # up — warm red
    if flav == 1:
        return wp.vec3(0.4, 0.62, 1.0)               # down — blue
    if flav == 2:
        return wp.vec3(1.0, 0.78, 0.3)               # charm — gold
    if flav == 3:
        return wp.vec3(0.55, 1.0, 0.6)               # strange — green
    if flav == 4:
        return wp.vec3(1.0, 0.5, 0.9)                # top — magenta
    return wp.vec3(0.6, 0.9, 1.0)                    # bottom — cyan


@wp.func
def color_charge(k: int) -> wp.vec3:
    """QCD colour charge red/green/blue — the three sum to colour-neutral."""
    if k == 0:
        return wp.vec3(1.0, 0.2, 0.2)
    if k == 1:
        return wp.vec3(0.25, 1.0, 0.3)
    return wp.vec3(0.3, 0.45, 1.0)
