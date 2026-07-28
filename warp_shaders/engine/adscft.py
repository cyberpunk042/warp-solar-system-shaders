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
* ``boundary_cft_dual`` — the SECOND copy of the CFT (complementary palette,
  counter-rotating flow): the other asymptotic boundary of the eternal black hole, the
  right factor of the thermofield-double state (Maldacena 2001; ER=EPR).
* ``rt_geodesic_glow`` — the bulk geodesic anchored on a boundary interval of the Poincaré
  disk: the Ryu-Takayanagi minimal surface AND (same curve in AdS₃) the static Wilson-loop
  string. Host-side quantitative dictionary: ``interval_entropy`` (Calabrese-Cardy /
  RT length), ``mutual_information`` (the min-pairing phase transition),
  ``string_turning_radius`` / ``screening_angle`` (where the quark string breaks on a
  horizon — deconfined screening).
* ``bh_entropy_evaporating`` / ``page_curve`` / ``page_time`` — the unitary Page curve as
  a minimum over gravitational saddles (Hawking vs island), crossing closed-form at
  ``t_page = T(1 − 2^{−3/2})`` (Penington; AEMM 2019).

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


@wp.func
def rt_geodesic_glow(zd: wp.vec2, th1: float, th2: float, px: float) -> float:
    """Glow of the bulk geodesic anchored at boundary angles th1, th2 (Poincaré disk).

    The unique circle through both endpoints orthogonal to the unit circle has centre
    c = (u + v)/(1 + u.v) and radius² = |c|² − 1; its arc inside the disk IS the
    hyperbolic geodesic — the RT minimal surface for the boundary interval AND the
    static Wilson-loop string in AdS₃ (both minimize proper length on the slice).
    """
    u = wp.vec2(wp.cos(th1), wp.sin(th1))
    v = wp.vec2(wp.cos(th2), wp.sin(th2))
    den = 1.0 + u[0] * v[0] + u[1] * v[1]
    c = wp.vec2((u[0] + v[0]) / den, (u[1] + v[1]) / den)
    rad = wp.sqrt(wp.max(wp.dot(c, c) - 1.0, 1.0e-8))
    darc = wp.abs(wp.length(zd - c) - rad)
    w = wp.max(0.0035, 1.5 * px)
    glow = wp.exp(-(darc * darc) / (w * w)) + 0.35 * wp.exp(-darc * 26.0)
    # endpoint dots — the boundary interval's operator insertions
    de = wp.min(wp.length(zd - u), wp.length(zd - v))
    glow += 2.2 * wp.exp(-(de * de) / (9.0 * w * w))
    return glow


@wp.func
def boundary_cft_dual(rd: wp.vec3, time: float, t_hawk: float) -> wp.vec3:
    """The SECOND copy of the CFT — the other boundary of the eternal black hole.

    The eternal Schwarzschild-AdS hole has two asymptotic regions, dual to TWO copies of
    the CFT entangled in the thermofield-double state |TFD⟩ = Σ e^{−βE/2}|E⟩_L|E⟩_R
    (Maldacena 2001). Same `{7,3}` fold, complementary palette and counter-rotating
    conformal flow — the L/R copies evolve with opposite time orientations in the TFD.
    """
    ph = -0.11 * time                      # counter-rotating: H_L = −H_R in the TFD
    cp = wp.cos(ph)
    sp = wp.sin(ph)
    d = wp.vec3(cp * rd[0] + sp * rd[2], rd[1], -sp * rd[0] + cp * rd[2])
    zs = wp.abs(d[1])
    sig = wp.vec2(d[0] / (1.0 + zs + 1.0e-6), d[2] / (1.0 + zs + 1.0e-6))
    s = 1.35 + 0.25 * wp.sin(0.17 * time + 2.6)
    folded = poincare_fold(sig * s)
    edge = tile_edge(folded, 0.0025)
    depth = wp.min(folded[2] / 10.0, 1.0)

    base = wp.vec3(0.007, 0.026, 0.052) * (0.6 + 1.2 * depth)
    lattice = wp.vec3(0.20, 0.66, 1.00) * edge * (0.55 + 1.0 * depth)
    pulse = 0.75 + 0.25 * wp.sin(2.2 * time + 5.0 * d[1] + 1.6)
    thermal = wp.vec3(0.10, 0.42, 1.00) * t_hawk * pulse * (0.35 + 0.65 * depth)
    return base + lattice + thermal


def interval_entropy(dtheta: float, eps: float = 0.05, c_central: float = 1.0) -> float:
    """Entanglement entropy of a boundary interval of angular size ``dtheta`` (host-side).

    Ryu-Takayanagi in global AdS₃ = the Calabrese-Cardy result for a CFT on a circle:
    S(A) = (c/3) ln( (2/ε) sin(Δθ/2) ) — the regularized length of the bulk geodesic
    hanging from the interval's endpoints, in units 4G = 1 (ε is the UV cutoff).
    Symmetric under Δθ → 2π − Δθ (a pure global state: S_A = S_Ā), maximal at Δθ = π.
    """
    return (c_central / 3.0) * math.log((2.0 / eps) * math.sin(0.5 * dtheta))


def mutual_information(gap: float, size_a: float, size_b: float,
                       eps: float = 0.05, c_central: float = 1.0):
    """Mutual information I(A:B) of two disjoint boundary intervals (host-side).

    A and B have angular sizes ``size_a``/``size_b`` separated by angular ``gap`` (the
    other gap is 2π − size_a − size_b − gap). RT computes S(A∪B) as the MINIMUM over the
    two allowed geodesic pairings — each interval capped by its own geodesic
    (*disconnected*) or the two gaps capped instead (*connected*):

        S_disc = S(size_a) + S(size_b)
        S_conn = S(gap) + S(2π − size_a − size_b − gap)

    I(A:B) = S_A + S_B − S(A∪B) is exactly 0 in the disconnected phase and jumps on with
    discontinuous first derivative at S_disc = S_conn — the holographic mutual-information
    phase transition. Returns ``(I, connected)``.
    """
    gap2 = 2.0 * math.pi - size_a - size_b - gap
    s_disc = interval_entropy(size_a, eps, c_central) + interval_entropy(size_b, eps, c_central)
    s_conn = interval_entropy(gap, eps, c_central) + interval_entropy(gap2, eps, c_central)
    connected = s_conn < s_disc
    return (max(s_disc - s_conn, 0.0), connected)


def string_turning_radius(dtheta: float, r_bdy: float) -> float:
    """Deepest bulk point of the geodesic string between two boundary points (host-side).

    The circle orthogonal to the boundary sphere of radius R through two points at
    half-angle α = Δθ/2 has centre distance d = R/cos α and radius ρ = R tan α
    (orthogonality: d² = R² + ρ²), so its closest approach to the origin is

        r_min = d − ρ = R (1 − sin α) / cos α

    → R as Δθ → 0 (a shallow string) and → 0 as Δθ → π (through the centre). In the
    black-hole phase the connected string exists only while r_min > r_h: once the
    geodesic would dip inside the horizon the string breaks into two radial segments
    ending ON the horizon — Debye screening of the quark pair in the deconfined plasma.
    """
    a = 0.5 * dtheta
    return r_bdy * (1.0 - math.sin(a)) / max(math.cos(a), 1.0e-12)


def screening_angle(r_h: float, r_bdy: float) -> float:
    """Quark separation Δθ at which the string breaks, inverting ``string_turning_radius``
    (host-side): r_min = R(1 − sin α)/cos α = r_h  ⇒  sin α = (R² − r_h²)/(R² + r_h²)."""
    s = (r_bdy * r_bdy - r_h * r_h) / (r_bdy * r_bdy + r_h * r_h)
    return 2.0 * math.asin(min(max(s, 0.0), 1.0))


def bh_entropy_evaporating(t: float, t_evap: float, s0: float = 1.0) -> float:
    """Bekenstein-Hawking entropy of an evaporating hole (host-side).

    Stefan-Boltzmann evaporation ``dM/dt ∝ −1/M²`` gives ``M(t) = M₀(1 − t/T)^{1/3}``,
    and ``S ∝ Area ∝ M²``, so ``S_BH(t) = S₀ (1 − t/T)^{2/3}`` — and the horizon radius
    scales as ``r_h(t) = r_h,0 (1 − t/T)^{1/3}``. Clamped to 0 after t_evap.
    """
    x = max(1.0 - t / t_evap, 0.0)
    return s0 * x ** (2.0 / 3.0)


def page_curve(t: float, t_evap: float, s0: float = 1.0):
    """The Page curve: radiation entropy as a minimum over two gravitational saddles
    (host-side). Returns ``(S_rad, island_dominant)``.

    * **Hawking saddle** (no island): ``S = S₀ − S_BH(t)`` — rises as radiation
      accumulates, and would rise forever (the information paradox).
    * **Island saddle**: the quantum extremal surface just inside the horizon puts the
      interior in the radiation's entanglement wedge, at the cost of the horizon area:
      ``S = S_BH(t)`` — falls as the hole evaporates.

    The gravitational path integral takes the MINIMUM (Penington; Almheiri-Engelhardt-
    Marolf-Maxfield 2019) — the same rule as ``mutual_information``'s pairing swap and
    the Hawking-Page ensemble: saddle competition. The result rises, turns over at the
    Page time, and returns to zero: unitarity restored.
    """
    s_bh = bh_entropy_evaporating(t, t_evap, s0)
    s_hawk = s0 - s_bh
    return (min(s_hawk, s_bh), s_hawk > s_bh)


def page_time(t_evap: float) -> float:
    """The Page time (host-side): the saddle crossing S₀ − S_BH = S_BH happens at
    S_BH = S₀/2, i.e. ``(1 − t/T)^{2/3} = 1/2`` ⇒ ``t_page = T (1 − 2^{−3/2})`` ≈ 0.6464 T."""
    return t_evap * (1.0 - 2.0 ** -1.5)


def horizon_radius(m: float, l_ads: float) -> float:
    """Horizon r_h of a Schwarzschild-AdS hole: the positive root of f(r) = 0 (bisection;
    f is monotone increasing past its minimum, host-side)."""
    lo, hi = 1.0e-6, 2.0 * m + l_ads
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if 1.0 + (mid * mid) / (l_ads * l_ads) - 2.0 * m / mid < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hawking_temperature(m: float, l_ads: float) -> float:
    """Hawking temperature T = f'(r_h)/4π of a Schwarzschild-AdS hole (host-side).

    T(r_h) = (L² + 3r_h²)/(4πL²r_h) has a minimum at r_h = L/√3: small AdS holes cool as
    they grow (negative specific heat, like flat space) while large ones heat up (positive
    specific heat) — the Hawking-Page structure.
    """
    r_h = horizon_radius(m, l_ads)
    fprime = 2.0 * r_h / (l_ads * l_ads) + 2.0 * m / (r_h * r_h)
    return fprime / (4.0 * math.pi)


def hawking_page_temperature(l_ads: float) -> float:
    """The Hawking-Page transition temperature T_HP = 1/(πL) (host-side).

    Below T_HP the canonical ensemble is dominated by *thermal AdS* (no black hole);
    above it, by the *large* black hole (whose horizon at the transition is r_h = L).
    On the boundary this is the confinement/deconfinement transition (Witten 1998).
    """
    return 1.0 / (math.pi * l_ads)


def large_hole_radius(t: float, l_ads: float) -> float:
    """Horizon of the LARGE (thermodynamically stable) hole at temperature t (host-side).

    Inverts T(r_h) = (L² + 3r_h²)/(4πL²r_h) on the large branch:
    r_h = (2πL²T + √(4π²L⁴T² − 3L²)) / 3. Only defined for t ≥ T_min = √3/(2πL);
    the discriminant is clamped so t just below T_min returns the minimum-T hole.
    """
    disc = 4.0 * (math.pi ** 2) * (l_ads ** 4) * (t ** 2) - 3.0 * (l_ads ** 2)
    return (2.0 * math.pi * (l_ads ** 2) * t + math.sqrt(max(disc, 0.0))) / 3.0


def mass_of_radius(r_h: float, l_ads: float) -> float:
    """Mass of the Schwarzschild-AdS hole with horizon r_h: M = r_h(1 + r_h²/L²)/2
    (from f(r_h) = 0, host-side)."""
    return 0.5 * r_h * (1.0 + (r_h * r_h) / (l_ads * l_ads))
