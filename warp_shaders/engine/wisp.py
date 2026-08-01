"""The wisp-in-the-box dictionary (host-side): trapped by geometry, exactly.

The brief: a wisp inside a box — the box is the boundary of the simulation — but the
wisp can never reach the corners, because it lives inside a spherical magic circle:
an AdS bubble. That is not a metaphor stapled onto physics; it IS the physics of
anti-de Sitter space, and every law below is exact and test-asserted (AdS radius
L = 1; global coordinates ``ds² = −(1+r²)dt² + dr²/(1+r²)``, Poincaré-disk radius
``r_disk = tanh(ρ/2)`` for proper distance ρ):

* ``proper_distance`` / ``disk_radius`` — the rim of the disk sits at FINITE map
  radius but INFINITE proper distance: ``ρ = 2·atanh(r_disk)`` diverges as
  ``r_disk → 1`` (asserted). However far the wisp flies, ``tanh(ρ/2) < 1``: the
  corners are visible and unreachable — trapped without walls.
* ``geodesic isochrony`` — AdS is a perfect harmonic trap: EVERY coasting body
  oscillates through the center with coordinate period 2π, INDEPENDENT of
  amplitude (asserted by integrating ``radial_geodesic`` at amplitudes 1.5 and
  4.0). Cut the engines anywhere and the bubble hands you back.
* ``hover_acceleration`` — to hang motionless at proper distance ρ takes proper
  acceleration ``a = tanh(ρ) < 1`` (asserted bounded): the engine required to
  hover NEVER exceeds c²/L. Any real drive can hold any altitude — hovering is
  cheap. Escape is not:
* ``static_energy`` — the energy of a unit-mass wisp at rest at ρ is
  ``E = cosh(ρ)``, and it DIVERGES (asserted): each step outward costs
  exponentially more. ``climb_energy(ρ₁→ρ₂) = cosh ρ₂ − cosh ρ₁`` is the fuel
  bill. The wisp that wants altitude must GROW and RETAIN energy — the boundary
  is not forbidden, it is infinitely expensive.
* ``radial_geodesic`` / ``radial_geodesic_closed`` — free fall through the
  bubble, twice: the RK4 integrator AND the exact closed form

      r(t) = √(E²−1)·sin t / √(E²cos²t + sin²t),   E = √(1+r_max²)

  (from proper-time SHM: ``r(τ) = √(E²−1)·sin τ``, ``t = atan(E·tan τ)``) —
  asserted to agree, and the closed form makes the 2π isochrony MANIFEST: the
  period never depends on E.

Stage 2 — the body the wisp grows into and hovers with — adds the orbit dictionary
(``ds² = −(1+r²)dt² + dr²/(1+r²) + r²dφ²``):

* ``orbit_angular_momentum`` — circular geodesics sit at ``V′ = 0`` exactly when
  ``L = r²`` (asserted), and they are STABLE (``V″ = 8``, radius-independent,
  asserted).
* ``orbit_energy`` — a body on a circular orbit at proper distance ρ carries
  ``E = 1 + r² = cosh²ρ`` (asserted): more than the ``cosh ρ`` of static hover —
  the kinetic surcharge — but it buys:
* ``orbit_angular_velocity`` — ``ω = (L/r²)(1+r²)/E = 1`` EXACTLY, at every
  radius (asserted at three radii): all circular orbits complete in 2π. Orbiting
  is FREE HOVER — zero thrust, forever — and the whole bubble turns in step.
* ``local_gravity`` — the g a hovering body feels is ``tanh ρ`` (the same as its
  hover thrust — the equivalence principle): a body released from rest falls
  ``½·tanh(ρ)·τ²`` in its own proper time (asserted against the closed-form
  geodesic). Retention runs the fuel bill in reverse: lowering mass from ρ₂ to
  ρ₁ BANKS ``cosh ρ₂ − cosh ρ₁`` — AdS is conservative; altitude is a battery.

Stage 3 — navigation — the cost algebra of getting anywhere in a space that
resists arrival. Moving forward in the simulation means using the drives; the
geometry then does something no Newtonian sky does — it makes every road the same
length in time:

* ``transfer_orbit`` — the ballistic (Hohmann-like) transfer between hover
  shells ρ₁ and ρ₂ is the geodesic whose apsides are the two shells, and its
  constants are pure hyperbolic algebra: ``E = cosh ρ₁ · cosh ρ₂``,
  ``L = sinh ρ₁ · sinh ρ₂`` (asserted: the effective potential equals E² at
  BOTH apsides).
* ``transfer_cost`` — leaving the ρ₁ orbit costs ``cosh ρ₁ (cosh ρ₂ − cosh ρ₁)``
  (the boost), circularizing at ρ₂ costs ``cosh ρ₂ (cosh ρ₂ − cosh ρ₁)``, and
  the total telescopes to ``cosh²ρ₂ − cosh²ρ₁`` — exactly
  ``orbit_energy(ρ₂) − orbit_energy(ρ₁)``: the bill is PATH-INDEPENDENT
  (asserted). AdS is conservative; there is no clever route, only the fare.
* ``the isochronous subway`` — EVERY transfer between ANY two shells takes
  coordinate time ``Δt = π/2`` and sweeps ``Δφ = π/2`` exactly (asserted): a
  quarter period, a quarter turn, however near or far. Timetables in the bubble
  are trivial; only the fare varies.
* ``apsides`` / ``geodesic_u`` — the general free arc, exactly: with u = r²,
  ``u(τ) = ū − A·cos 2τ`` where ``ū = (E²−1−L²)/2``, ``A = √(ū²−L²)`` (SHM in
  u; asserted against the coordinate-time integrator), apsides at ``ū ± A``.
* ``the geodesic lens`` — launch test motes from one point in ANY direction
  with ANY speed: ALL of them reconverge at the antipodal point at ``t = π``
  and return HOME at ``t = 2π`` (asserted for three different (E, L) at 1e-6).
  In the bubble you cannot get lost — only be early with more fuel.

Stage 1 in 3D — the ball. The magic circle becomes a magic SPHERE (the Poincaré
ball, the spatial slice of global AdS₄), and every stage-1 law carries over
UNCHANGED, because the radial equation never mentioned dimension: the rim is still
at ``tanh(ρ/2) < 1``, the isochrony is still 2π, the fuel wall is still ``cosh ρ``,
and every free flight is planar (angular momentum is conserved), so the closed
forms apply verbatim in each geodesic's own plane. What 3D adds is the SIZE of
the trap, and it is monstrous:

* ``sphere_area`` — the geodesic sphere at proper radius ρ has area
  ``A = 4π sinh²ρ`` — EXPONENTIAL growth (asserted: A(ρ+1)/A(ρ) → e²); the
  Euclidean ``4πρ²`` only survives as the small-ρ limit (asserted).
* ``ball_volume`` — the ball's volume is ``V = π(sinh 2ρ − 2ρ)`` — exactly the
  integral of the area (``dV/dρ = A``, asserted), Euclidean ``4πρ³/3`` at small
  ρ (asserted).
* ``volume_area_ratio`` — ``V/A → 1/2`` as ρ → ∞ (asserted): however huge the
  ball grows, essentially ALL of its volume lies within ONE unit of its surface.
  Hyperbolic space is all skin and no core — the geometric seed of holography:
  the bulk lives at its boundary.
* ``the isochronous firework`` — release motes from the center in EVERY
  direction with EVERY amplitude: each follows the same closed form along its
  own ray, so the whole swarm passes through r = 0 SIMULTANEOUSLY every π
  (asserted): the explosion that un-explodes, twice per period.

The boundary sees everything — the shadow. The brief's reason for the trap was
"because of AdS/CFT": the bubble has a boundary theory, and the wisp casts an
exact shadow on it. The bulk-to-boundary propagator in global AdS₃ for a boundary
operator of dimension Δ is

    K(ρ, θ) = (cosh ρ − sinh ρ · cos θ)^(−Δ)

(θ = boundary angle from the wisp's direction), and three exact laws follow:

* ``shadow_contrast`` — the shadow's peak-to-antipode contrast is ``e^{2Δρ}``
  EXACTLY (asserted at machine precision): the higher the wisp climbs, the
  sharper the boundary knows where it is — exponentially.
* ``shadow_width`` — the half-max angular width has the closed form
  ``θ_½ = acos[(cosh ρ − 2^{1/Δ} e^{−ρ})/sinh ρ]`` (asserted against numeric
  half-max), with ``θ_½ · e^ρ → 2√(2^{1/Δ}−1)`` (asserted): the UV/IR
  correspondence, quantified — bulk depth IS boundary resolution.
* ``the conserved imprint`` — for Δ = 1 the TOTAL shadow ``∫K dθ = 2π`` for
  EVERY ρ (asserted at 10⁻⁶): climbing concentrates the imprint but never
  changes its total. The boundary never loses track of the wisp; it cannot.
  Holography is not surveillance added to the bubble — it is the bubble.

Stage 1: coast (the trap always returns you), burn (the drive climbs, the map
compresses), fall back (the fuel wall wins). Stage 2: grow, climb, hover on flame,
tip into orbit — the body holds altitude for free. Stage 3: boost, ride the
quarter-period arc, circularize — and watch the lens refocus everything you
release. And through all of it, the shadow on the rim tells the boundary exactly
where the wisp is. See ``docs/research/57-the-wisp-in-the-box.md``.
"""

import math


def proper_distance(r_disk: float) -> float:
    """Proper distance from the center to Poincaré-disk radius r (host-side):
    ``ρ = 2·atanh(r)`` — DIVERGES as r → 1 (asserted): the rim is infinitely far."""
    return 2.0 * math.atanh(r_disk)


def disk_radius(rho: float) -> float:
    """Where proper distance ρ lands on the map (host-side): ``r = tanh(ρ/2)`` —
    always < 1 (asserted): however far the wisp flies, the rim never arrives."""
    return math.tanh(0.5 * rho)


def hover_acceleration(rho: float) -> float:
    """Proper acceleration needed to hang motionless at proper distance ρ
    (host-side): ``a = tanh(ρ)`` — BOUNDED by c²/L = 1 (asserted): hovering
    anywhere is within any real engine's reach."""
    return math.tanh(rho)


def static_energy(rho: float) -> float:
    """Energy of a unit-mass wisp at rest at proper distance ρ (host-side):
    ``E = cosh(ρ)`` — diverges (asserted): altitude is paid for exponentially.
    This is the wall the wisp's fuel meets."""
    return math.cosh(rho)


def climb_energy(rho1: float, rho2: float) -> float:
    """The fuel bill (host-side): ``ΔE = cosh(ρ₂) − cosh(ρ₁)`` — the energy a
    unit-mass wisp must spend (and therefore first retain) to climb from ρ₁ to
    ρ₂. Doubling altitude far out costs ~e^ρ more."""
    return math.cosh(rho2) - math.cosh(rho1)


def radial_geodesic(r_max: float, t_span: float, n: int = 40000):
    """Integrate the exact free-fall through the bubble (host-side, RK4 on
    ``dr/dt = ±√(E²−(1+r²))·(1+r²)/E``, global AdS r = sinh ρ): returns
    (t_list, r_list). EVERY amplitude oscillates with coordinate period 2π —
    the isochrony the suite asserts at r_max = 1.5 and 4.0."""
    e = math.sqrt(1.0 + r_max * r_max)
    h = t_span / float(n)
    r, sgn = 0.0, 1.0
    ts = [0.0]
    rs = [0.0]

    def f(rr, sg):
        v2 = e * e - (1.0 + rr * rr)
        if v2 < 0.0:
            v2 = 0.0
        return sg * math.sqrt(v2) * (1.0 + rr * rr) / e

    for k in range(n):
        k1 = f(r, sgn)
        k2 = f(r + 0.5 * h * k1, sgn)
        k3 = f(r + 0.5 * h * k2, sgn)
        k4 = f(r + h * k3, sgn)
        r = r + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if r >= r_max:
            r, sgn = r_max - 1e-12, -1.0
        elif r <= -r_max:
            r, sgn = -r_max + 1e-12, 1.0
        ts.append(h * float(k + 1))
        rs.append(r)
    return ts, rs


def radial_geodesic_closed(r_max: float, t: float) -> float:
    """The EXACT radial free-fall (host-side): ``r(t) = √(E²−1)·sin t /
    √(E²cos²t + sin²t)`` with ``E = √(1+r_max²)`` — derived from proper-time SHM
    (``r(τ) = √(E²−1)·sin τ``, ``t = atan(E tan τ)``); asserted against the RK4
    integrator. Period 2π for every amplitude, by inspection: isochrony."""
    e = math.sqrt(1.0 + r_max * r_max)
    st, ct = math.sin(t), math.cos(t)
    return r_max * st / math.sqrt(e * e * ct * ct + st * st)


def orbit_angular_momentum(r: float) -> float:
    """Angular momentum of the circular orbit at global radius r (host-side):
    ``L = r²`` — the exact stationary point of ``V² = (1+r²)(1+L²/r²)``
    (``V′ = 0`` asserted; ``V″ = 8 > 0``: stable at every radius, asserted)."""
    return r * r


def orbit_energy(rho: float) -> float:
    """Energy of a unit-mass body on a circular orbit at proper distance ρ
    (host-side): ``E = 1 + r² = cosh²ρ`` (asserted) — the kinetic surcharge over
    static hover's ``cosh ρ``, paid once, and then the thrust bill is zero."""
    return math.cosh(rho) ** 2


def orbit_angular_velocity(r: float) -> float:
    """The universal clock (host-side): ``ω = (L/r²)·(1+r²)/E = 1`` EXACTLY for
    every circular orbit (asserted at three radii): the whole bubble orbits in
    step, one lap per 2π — free hover with the same isochrony as free fall."""
    ell = orbit_angular_momentum(r)
    e = math.sqrt((1.0 + r * r) * (1.0 + ell * ell / (r * r)))
    return (ell / (r * r)) * (1.0 + r * r) / e


def local_gravity(rho: float) -> float:
    """The g a hovering body feels at proper distance ρ (host-side): ``tanh ρ`` —
    identical to its hover thrust (the equivalence principle): released from
    rest it falls ``½·tanh(ρ)·τ²`` in proper time (asserted against the exact
    geodesic)."""
    return math.tanh(rho)


def sphere_area(rho: float) -> float:
    """Area of the geodesic sphere at proper radius ρ in the 3D ball
    (host-side): ``A = 4π sinh²ρ`` — EXPONENTIAL (asserted A(ρ+1)/A(ρ) → e²);
    the Euclidean 4πρ² survives only as the small-ρ limit (asserted)."""
    s = math.sinh(rho)
    return 4.0 * math.pi * s * s


def ball_volume(rho: float) -> float:
    """Volume of the ball of proper radius ρ (host-side):
    ``V = π(sinh 2ρ − 2ρ)`` — exactly the integral of ``sphere_area``
    (dV/dρ = A, asserted); Euclidean 4πρ³/3 at small ρ (asserted)."""
    return math.pi * (math.sinh(2.0 * rho) - 2.0 * rho)


def volume_area_ratio(rho: float) -> float:
    """The skin theorem (host-side): ``V/A → 1/2`` as ρ → ∞ (asserted) —
    however huge the ball, essentially all of its volume lies within one unit
    of its surface. Hyperbolic space is all skin and no core; the bulk lives
    at its boundary."""
    return ball_volume(rho) / sphere_area(rho)


def shadow_kernel(rho: float, theta: float, delta: float = 1.0) -> float:
    """The wisp's boundary shadow (host-side): the exact bulk-to-boundary
    propagator of global AdS₃, ``K = (cosh ρ − sinh ρ cos θ)^{−Δ}`` — θ measured
    from the wisp's direction. Uniform at ρ = 0; a spike as ρ grows."""
    return (math.cosh(rho) - math.sinh(rho) * math.cos(theta)) ** (-delta)


def shadow_contrast(rho: float, delta: float = 1.0) -> float:
    """Peak-to-antipode contrast of the shadow (host-side): ``e^{2Δρ}`` EXACTLY
    (asserted at machine precision against the kernel) — the boundary's
    knowledge of the wisp's position sharpens exponentially with altitude."""
    return math.exp(2.0 * delta * rho)


def shadow_width(rho: float, delta: float = 1.0) -> float:
    """Half-max angular width of the shadow (host-side), closed form:
    ``θ_½ = acos[(cosh ρ − 2^{1/Δ} e^{−ρ})/sinh ρ]`` (asserted against the
    numeric half-max), with ``θ_½ · e^ρ → 2√(2^{1/Δ}−1)`` (asserted) — the
    UV/IR correspondence: bulk depth IS boundary resolution. Returns π (fully
    spread) when the kernel never falls to half peak."""
    sh = math.sinh(rho)
    if sh < 1e-12:
        return math.pi
    c = (math.cosh(rho) - 2.0 ** (1.0 / delta) * math.exp(-rho)) / sh
    if c < -1.0:
        return math.pi
    return math.acos(min(c, 1.0))


def transfer_orbit(rho1: float, rho2: float):
    """The ballistic transfer between hover shells ρ₁ and ρ₂ (host-side): the
    geodesic whose apsides ARE the two shells. Pure hyperbolic algebra:
    ``E = cosh ρ₁ · cosh ρ₂``, ``L = sinh ρ₁ · sinh ρ₂`` (asserted: the
    effective potential equals E² at both apsides). Returns (E, L)."""
    return (math.cosh(rho1) * math.cosh(rho2),
            math.sinh(rho1) * math.sinh(rho2))


def transfer_cost(rho1: float, rho2: float):
    """The fare (host-side): boost off the ρ₁ orbit, coast, circularize at ρ₂.
    Returns (boost, circularize, total) with
    ``boost = cosh ρ₁ (cosh ρ₂ − cosh ρ₁)``,
    ``circularize = cosh ρ₂ (cosh ρ₂ − cosh ρ₁)``, and the total telescoping to
    ``cosh²ρ₂ − cosh²ρ₁ = orbit_energy(ρ₂) − orbit_energy(ρ₁)`` — the bill is
    PATH-INDEPENDENT (asserted): no clever route exists, only the fare."""
    d = math.cosh(rho2) - math.cosh(rho1)
    boost = math.cosh(rho1) * d
    circ = math.cosh(rho2) * d
    return boost, circ, boost + circ


def apsides(e: float, ell: float):
    """Turning radii of the free arc with constants (E, L) (host-side): with
    u = r², the radial equation is SHM in u about ``ū = (E²−1−L²)/2`` with
    amplitude ``A = √(ū²−L²)``; the apsides are ``r_∓ = √(ū ∓ A)``. Returns
    (r_min, r_max). Circular orbits are the degenerate case A = 0."""
    ubar = 0.5 * (e * e - 1.0 - ell * ell)
    amp2 = ubar * ubar - ell * ell
    amp = math.sqrt(amp2) if amp2 > 0.0 else 0.0
    return math.sqrt(max(ubar - amp, 0.0)), math.sqrt(ubar + amp)


def geodesic_u(e: float, ell: float, tau: float) -> float:
    """The EXACT free arc (host-side): ``u(τ) = ū − A·cos 2τ`` (u = r², proper
    time τ from periapsis) — SHM in u, period π, asserted against the
    coordinate-time integrator. Every bound arc breathes at the same rate."""
    ubar = 0.5 * (e * e - 1.0 - ell * ell)
    amp2 = ubar * ubar - ell * ell
    amp = math.sqrt(amp2) if amp2 > 0.0 else 0.0
    return ubar - amp * math.cos(2.0 * tau)


def orbit_geodesic(e: float, ell: float, r0: float, phi0: float, sgn: float,
                   t_span: float, n: int = 40000):
    """Integrate the free arc in COORDINATE time (host-side, RK4 on
    ``dr/dt = ±√(E²−V²)·(1+r²)/E``, ``dφ/dt = (L/r²)(1+r²)/E``): returns
    (t_list, r_list, phi_list). This is the integrator the suite uses to assert
    the π/2 subway, the closed form ``geodesic_u``, and the geodesic lens
    (antipodal refocus at t = π, home at t = 2π)."""
    r_min, r_max = apsides(e, ell)
    h = t_span / float(n)
    r, phi, sg = r0, phi0, sgn
    ts, rs, phis = [0.0], [r0], [phi0]

    def fr(rr, s):
        v2 = e * e - (1.0 + rr * rr) * (1.0 + ell * ell / (rr * rr))
        if v2 < 0.0:
            v2 = 0.0
        return s * math.sqrt(v2) * (1.0 + rr * rr) / e

    def fp(rr):
        return (ell / (rr * rr)) * (1.0 + rr * rr) / e

    for k in range(n):
        k1, p1 = fr(r, sg), fp(r)
        k2, p2 = fr(r + 0.5 * h * k1, sg), fp(r + 0.5 * h * k1)
        k3, p3 = fr(r + 0.5 * h * k2, sg), fp(r + 0.5 * h * k2)
        k4, p4 = fr(r + h * k3, sg), fp(r + h * k3)
        r = r + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        phi = phi + (h / 6.0) * (p1 + 2.0 * p2 + 2.0 * p3 + p4)
        if r >= r_max:
            r, sg = r_max - 1e-12, -1.0
        elif r <= r_min:
            r, sg = r_min + 1e-12, 1.0
        ts.append(h * float(k + 1))
        rs.append(r)
        phis.append(phi)
    return ts, rs, phis


def geodesic_period(r_max: float) -> float:
    """Measure the coasting period off the exact integration (host-side): time
    between successive upward zero crossings — asserted ≈ 2π independent of
    amplitude: the bubble is an isochronous trap."""
    ts, rs = radial_geodesic(r_max, 15.0, 60000)
    crossings = []
    for k in range(1, len(rs)):
        if rs[k - 1] < 0.0 <= rs[k]:
            # linear interpolation of the crossing time
            f = -rs[k - 1] / (rs[k] - rs[k - 1])
            crossings.append(ts[k - 1] + f * (ts[k] - ts[k - 1]))
        if len(crossings) == 2:
            return crossings[1] - crossings[0]
    raise RuntimeError("no full oscillation found — increase t_span")
