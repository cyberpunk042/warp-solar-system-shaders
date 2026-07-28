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
* ``lloyd_bound`` / ``complexity_rate`` / ``complexity_growth`` / ``scrambling_time`` —
  holographic complexity: growth saturating ``dC/dt = 2M/π`` from below (CA late-time
  rate = the Lloyd bound), linear-forever interior growth, ``t_* = (β/2π)·ln S``.
* The **BTZ dictionary** (all exact — 3D gravity's gift): ``btz_horizon_radius`` /
  ``btz_temperature`` / ``btz_entropy`` (``r_h = L√M``, ``T = r_h/2πL²``,
  ``S = 2πr_h/4G``), ``horizon_translation_length`` / ``quotient_wall_position`` (the
  hole as a quotient of the disk), ``thermal_interval_entropy`` /
  ``thermal_entanglement`` / ``plateau_angle`` (finite-T RT with homology: the
  entanglement plateau, Araki-Lieb saturated exactly), ``btz_qnm`` (exact quasinormal
  modes ``ω = ±k − 4πiT(n + Δ/2)`` = the poles of the dual thermal correlator).
* ``in_boundary_arc`` / ``geodesic_far_side`` — device helpers for interval bands and
  entanglement-wedge shading on the disk.

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


@wp.func
def in_boundary_arc(th: float, t1: float, t2: float) -> float:
    """1 when boundary angle th lies in the ccw arc [t1, t2], else 0 (wrap-safe)."""
    span = t2 - t1 - 6.2831853 * wp.floor((t2 - t1) / 6.2831853)
    d = th - t1 - 6.2831853 * wp.floor((th - t1) / 6.2831853)
    out = float(0.0)
    if d < span:
        out = 1.0
    return out


@wp.func
def geodesic_far_side(zd: wp.vec2, th1: float, th2: float) -> float:
    """Soft mask: 1 on the far side of the geodesic capping the ccw arc [th1, th2].

    The orthogonal circle |z − c| = rad separates the disk into an arc side and a far
    side; which sign is which flips when the arc subtends more than π, so it is fixed
    by the arc midpoint m (always on the arc side): far = the sign of (|z − c| − rad)
    OPPOSITE to (|m − c| − rad)."""
    u = wp.vec2(wp.cos(th1), wp.sin(th1))
    v = wp.vec2(wp.cos(th2), wp.sin(th2))
    den = 1.0 + u[0] * v[0] + u[1] * v[1] + 1.0e-6
    c = wp.vec2((u[0] + v[0]) / den, (u[1] + v[1]) / den)
    rad = wp.sqrt(wp.max(wp.dot(c, c) - 1.0, 1.0e-8))
    span = th2 - th1 - 6.2831853 * wp.floor((th2 - th1) / 6.2831853)
    mth = th1 + 0.5 * span
    m = wp.vec2(wp.cos(mth), wp.sin(mth))
    flip = 1.0
    if wp.length(m - c) - rad > 0.0:
        flip = -1.0
    return wp.clamp(flip * (wp.length(zd - c) - rad) / 0.02, 0.0, 1.0)


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


def lloyd_bound(m: float) -> float:
    """The Lloyd bound on complexity growth (host-side): ``dC/dt ≤ 2M/π`` (ħ=1).

    Brown-Roberts-Susskind-Swingle-Zhao: the late-time action growth of the eternal
    AdS black hole's Wheeler-DeWitt patch is exactly ``2M/π`` — black holes saturate
    the fastest-computer bound nature allows.
    """
    return 2.0 * m / math.pi


def complexity_rate(t: float, m: float, t_ramp: float) -> float:
    """Growth rate of holographic complexity (host-side model interpolation):
    ``dC/dt = (2M/π)·tanh(t/t_ramp)`` — zero at t=0, rising monotonically, and
    approaching the Lloyd bound FROM BELOW at late times (never exceeding it).

    The exact late-time rate is the Lloyd bound; the smooth ramp is a documented
    visualization interpolation with the qualitatively correct shape (Carmi et al.
    1709.10184: the rate rises from zero and asymptotes).
    """
    return lloyd_bound(m) * math.tanh(t / t_ramp)


def complexity_growth(t: float, m: float, t_ramp: float) -> float:
    """Holographic complexity C(t) of the eternal hole (host-side): the integral of
    ``complexity_rate`` — ``C(t) = (2M/π)·t_ramp·ln cosh(t/t_ramp)``. Quadratic at
    early times, LINEAR WITHOUT END at late times: the boundary state looks thermal
    after a few thermal times, but the Einstein-Rosen interior — and the complexity
    dual to its volume — keeps growing for exponentially long (Susskind's
    complexity=volume). Numerically stable for large ``t/t_ramp``.
    """
    x = t / t_ramp
    lncosh = x - math.log(2.0) if x > 20.0 else math.log(math.cosh(x))
    return lloyd_bound(m) * t_ramp * lncosh


def scrambling_time(s: float, t_hawk: float) -> float:
    """The fast-scrambling time (host-side): ``t_* = (β/2π)·ln S`` — black holes
    scramble information in a time logarithmic in their entropy (Sekino-Susskind),
    the fastest scramblers in nature. β = 1/T_Hawking."""
    return math.log(s) / (2.0 * math.pi * t_hawk)


def btz_horizon_radius(m: float, l_ads: float) -> float:
    """BTZ horizon radius (host-side): in 2+1 dimensions the blackening factor is
    ``f(r) = −M + r²/L²`` (no 1/r term — no Newtonian tail in 3D gravity), so
    ``r_h = L·√M`` exactly. The BTZ hole (Bañados-Teitelboim-Zanelli 1992) is locally
    pure AdS₃ everywhere: all of its physics is in the global identification."""
    return l_ads * math.sqrt(m)


def btz_temperature(r_h: float, l_ads: float) -> float:
    """BTZ Hawking temperature (host-side): ``T = f'(r_h)/4π = r_h/(2πL²)`` — LINEAR in
    the horizon radius, so BTZ holes always have positive specific heat (no small/large
    branch structure; the 3D box is a perfect thermal cavity)."""
    return r_h / (2.0 * math.pi * l_ads * l_ads)


def btz_entropy(r_h: float) -> float:
    """BTZ Bekenstein-Hawking entropy (host-side): the 'area' of a 2+1-dimensional
    horizon is its LENGTH — ``S = 2πr_h/(4G)`` (4G = 1 here). This is the entropy
    Strominger reproduced exactly from the Cardy formula of the boundary CFT."""
    return 2.0 * math.pi * r_h / 4.0


def horizon_translation_length(r_h: float, l_ads: float) -> float:
    """The quotient's translation length (host-side): BTZ is AdS₃ / Γ where Γ is
    generated by ONE hyperbolic isometry. On the t = 0 Poincaré-disk slice the isometry
    translates along a geodesic axis by hyperbolic distance ``λ = 2πr_h/L``, and the
    axis projects to the CLOSED horizon geodesic of the quotient — the horizon *is* the
    translation length: ``S = λ·L/(4G)``. Entropy is literally a length of the disk."""
    return 2.0 * math.pi * r_h / l_ads


def quotient_wall_position(n: int, lam: float) -> float:
    """Where the n-th image of the fundamental-domain wall crosses the axis (host-side).

    Put the axis on the disk's x-axis (fixed points ±1). The generator moves axis points
    by hyperbolic distance λ, so the n-th wall crosses at hyperbolic distance n·λ from
    the origin: ``x_n = tanh(n·λ/2)`` — the walls accumulate at the fixed points ±1,
    which is what a quotient by a hyperbolic element looks like."""
    return math.tanh(0.5 * float(n) * lam)


def thermal_interval_entropy(theta: float, temp: float, eps: float = 0.05,
                             c_central: float = 1.0) -> float:
    """Entanglement entropy of a boundary interval at temperature T (host-side).

    RT in the BTZ background = Calabrese-Cardy at finite temperature:
    ``S(θ) = (c/3)·ln( (β/(πε))·sinh(πθ/β) )`` with β = 1/T. Two honest limits,
    both test-asserted:

    * ``θ ≪ β``: ``S → (c/3)·ln(θ/ε)`` — the vacuum answer (UV physics doesn't feel T);
    * ``θ ≫ β``: ``S → (πc/3)·T·θ`` — EXTENSIVE, slope = the thermal entropy density:
      the interval stops learning geometry and just counts thermal excitations.
    """
    beta = 1.0 / temp
    return (c_central / 3.0) * math.log(
        (beta / (math.pi * eps)) * math.sinh(math.pi * theta / beta))


def thermal_entanglement(theta: float, temp: float, s_bh: float, eps: float = 0.05,
                         c_central: float = 1.0):
    """Entanglement entropy of an interval on the BTZ boundary WITH the homology
    constraint (host-side). Returns ``(S, wrapped)``.

    Two RT candidates compete — the third saddle competition of the set:

    * **direct**: the geodesic hugging the interval — ``S = S_th(θ)``;
    * **wrapped**: the complement's geodesic PLUS the horizon (homology forces the
      surface to wrap it) — ``S = S_th(2π − θ) + S_BH``.

    The minimum switches at the **entanglement plateau**: past the plateau angle the
    entropy saturates at the wrapped value and the interval's wedge swallows the black
    hole. In the plateau, Araki-Lieb ``|S(A) − S(Ā)| ≤ S_BH`` is EXACTLY saturated.
    """
    s_direct = thermal_interval_entropy(theta, temp, eps, c_central)
    s_wrap = thermal_interval_entropy(2.0 * math.pi - theta, temp, eps, c_central) + s_bh
    return (min(s_direct, s_wrap), s_wrap < s_direct)


def plateau_angle(temp: float, s_bh: float, eps: float = 0.05,
                  c_central: float = 1.0) -> float:
    """The entanglement-plateau angle (host-side): the θ where the direct and wrapped
    RT candidates exchange dominance — ``S_th(θ*) = S_th(2π − θ*) + S_BH``. The
    difference is monotone in θ, so bisection converges; returns 2π if the plateau is
    never reached (S_BH too large for this temperature)."""
    def gap(th: float) -> float:
        return (thermal_interval_entropy(th, temp, eps, c_central)
                - thermal_interval_entropy(2.0 * math.pi - th, temp, eps, c_central) - s_bh)
    lo, hi = math.pi, 2.0 * math.pi - 1.0e-6
    if gap(hi) < 0.0:
        return 2.0 * math.pi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if gap(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def btz_qnm(k: float, n: int, temp: float, delta: float = 2.0):
    """BTZ quasinormal frequencies (host-side) — EXACT, one of the very few black holes
    whose ringdown is known in closed form (Birmingham-Sachs-Solodukhin 2001):

        ``ω = ±k − 4πi·T·(n + Δ/2)``

    for a field dual to a boundary operator of dimension Δ, momentum k, overtone n.
    Returns ``(ω_re, ω_im)`` with ω_im < 0: every perturbation rings at the momentum
    frequency and decays at a rate set ONLY by the temperature — and on the boundary
    this IS the thermalization rate of the dual CFT (the poles of the retarded thermal
    correlator). Overtones are spaced exactly 4πT apart."""
    return (k, -4.0 * math.pi * temp * (float(n) + 0.5 * delta))


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
