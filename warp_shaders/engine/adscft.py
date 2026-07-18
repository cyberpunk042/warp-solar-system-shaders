"""Shared AdS/CFT holography helpers (device ``@wp.func`` + host-side physics).

The engine-level core behind the holography set:

* ``poincare_fold`` — the `{7,3}` hyperbolic reflection-group fold on the Poincaré disk,
  returning the folded point, the reflection count (orbit trap / generation) and the
  accumulated conformal magnification (for constant screen-width anti-aliasing). Used by
  the ``ads_cft`` disk scene AND as the conformal-lattice texture painted on the boundary
  of the ray-traced bulk (``ads_bulk``) — one fold, two duals.
* ``ads_blackening`` — the Schwarzschild-AdS blackening factor ``f(r) = 1 + r²/L² − 2M/r``:
  the gravitational redshift of the bulk, diverging at the conformal boundary (the CFT's
  UV). Rendering normalizes against a finite cutoff radius — literally *holographic
  renormalization*.
* ``boundary_cft`` — the CFT painted on the conformal boundary sphere: the hyperbolic
  lattice stereographically projected onto the boundary, with a thermal wash set by the
  black hole's **Hawking temperature** (host-side ``hawking_temperature``) — the bulk hole
  IS a thermal state of the boundary theory (Hawking & Page 1983).

Null-geodesic honesty: in Schwarzschild-AdS the photon orbital equation is
``d²u/dφ² + u = 3Mu²`` — the cosmological constant **drops out of the path shape**
(Islam 1983), so the bulk integrator reuses the proven Schwarzschild pull
``a = −(3/2) h² x/r⁵`` of ``engine.blackhole``; what AdS adds is the *timelike conformal
boundary at finite optical distance* that reflects light back in (the "AdS box"), handled
by the scene's bounce loop. See ``docs/research/46-ads-cft-holography.md``.
"""

import math

import warp as wp

# ---- {p,q} = {7,3} tiling constants (derived, not tuned) -----------------------------------
# Right hyperbolic triangle O-M-V (polygon centre, edge midpoint, vertex) with angles
# pi/p at O and pi/q at V:  cosh(OM) = cos(pi/q) / sin(pi/p).
_P = 7.0
_Q = 3.0
_A = math.pi / _P
_COSH_M = math.cos(math.pi / _Q) / math.sin(math.pi / _P)
_M = math.acosh(_COSH_M)
_X0 = math.tanh(0.5 * _M)

WEDGE = wp.constant(_A)
WEDGE2 = wp.constant(2.0 * _A)
MIR_D = wp.constant((1.0 + _X0 * _X0) / (2.0 * _X0))   # mirror centre (orthogonal: d² = 1 + r²)
MIR_R2 = wp.constant(((1.0 + _X0 * _X0) / (2.0 * _X0) - _X0) ** 2)
FOLDS = 48


@wp.func
def poincare_fold(z: wp.vec2) -> wp.vec4:
    """Fold z into the {7,3} fundamental domain.

    Returns ``vec4(z'.x, z'.y, depth, scale)``: the folded point, the number of mirror
    inversions (the tiling "generation" — an orbit trap), and the accumulated conformal
    magnification so callers can draw tile edges at constant *screen* width.
    """
    depth = float(0.0)
    scale = float(1.0)
    for _f in range(FOLDS):
        ang = wp.atan2(z[1], z[0])
        k = wp.floor((ang + WEDGE) / WEDGE2)
        if k != 0.0:
            ca = wp.cos(-k * WEDGE2)
            sa = wp.sin(-k * WEDGE2)
            z = wp.vec2(ca * z[0] - sa * z[1], sa * z[0] + ca * z[1])
        if z[1] < 0.0:
            z = wp.vec2(z[0], -z[1])
        w = wp.vec2(z[0] - MIR_D, z[1])
        r2 = wp.dot(w, w)
        if r2 < MIR_R2:
            kinv = MIR_R2 / r2
            z = wp.vec2(MIR_D + w[0] * kinv, w[1] * kinv)
            scale = scale * kinv
            depth += 1.0
        else:
            break
    return wp.vec4(z[0], z[1], depth, scale)


@wp.func
def tile_edge(folded: wp.vec4, px: float) -> float:
    """Anti-aliased tile-edge weight from a ``poincare_fold`` result at pixel size ``px``."""
    w = wp.vec2(folded[0] - MIR_D, folded[1])
    e = wp.abs(wp.length(w) - wp.sqrt(MIR_R2))
    npix = e / wp.max(folded[3] * px, 1.0e-12)
    return wp.exp(-0.5 * (npix / 1.5) * (npix / 1.5))


@wp.func
def ads_blackening(r: float, l_ads: float, m: float) -> float:
    """Schwarzschild-AdS blackening factor f(r) = 1 + r²/L² − 2M/r."""
    return 1.0 + (r * r) / (l_ads * l_ads) - 2.0 * m / r


@wp.func
def boundary_cft(rd: wp.vec3, time: float, t_hawk: float) -> wp.vec3:
    """The CFT on the conformal boundary, sampled in the direction ``rd`` of the hit point.

    Stereographic projection of the boundary sphere onto a plane is a conformal map, so the
    hyperbolic lattice drawn there IS a conformal field pattern on the boundary. The lattice
    slowly rotates (a boundary conformal flow), and a thermal wash scales with the bulk
    hole's Hawking temperature ``t_hawk`` — the thermal state dual to the black hole.
    """
    # slow conformal flow of the boundary pattern
    ph = 0.11 * time
    cp = wp.cos(ph)
    sp = wp.sin(ph)
    d = wp.vec3(cp * rd[0] + sp * rd[2], rd[1], -sp * rd[0] + cp * rd[2])
    # stereographic projection from the pole opposite the sample hemisphere (conformal)
    zs = wp.abs(d[1])
    sig = wp.vec2(d[0] / (1.0 + zs + 1.0e-6), d[2] / (1.0 + zs + 1.0e-6))
    # breathe the conformal scale so cells drift through generations
    s = 1.35 + 0.25 * wp.sin(0.17 * time)
    folded = poincare_fold(sig * s)
    edge = tile_edge(folded, 0.0025)
    depth = wp.min(folded[2] / 10.0, 1.0)

    base = wp.vec3(0.050, 0.026, 0.009) * (0.6 + 1.2 * depth)
    lattice = wp.vec3(1.00, 0.62, 0.22) * edge * (0.55 + 1.0 * depth)
    # thermal wash — the boundary theory heated to the hole's Hawking temperature
    pulse = 0.75 + 0.25 * wp.sin(2.2 * time + 5.0 * d[1])
    thermal = wp.vec3(1.00, 0.36, 0.10) * t_hawk * pulse * (0.35 + 0.65 * depth)
    return base + lattice + thermal


def hawking_temperature(m: float, l_ads: float) -> float:
    """Hawking temperature T = f'(r_h)/4π of a Schwarzschild-AdS hole (host-side).

    The horizon r_h is the positive root of f(r) = 1 + r²/L² − 2M/r (bisection; f is
    monotone increasing past its minimum). T(r_h) = (L² + 3r_h²)/(4πL²r_h) has a minimum
    at r_h = L/√3: small AdS holes cool as they grow (negative specific heat, like flat
    space) while large ones heat up (positive specific heat) — the Hawking-Page structure.
    """
    lo, hi = 1.0e-6, 2.0 * m + l_ads
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 1.0 + (mid * mid) / (l_ads * l_ads) - 2.0 * m / mid < 0.0:
            lo = mid
        else:
            hi = mid
    r_h = 0.5 * (lo + hi)
    fprime = 2.0 * r_h / (l_ads * l_ads) + 2.0 * m / (r_h * r_h)
    return fprime / (4.0 * math.pi)
