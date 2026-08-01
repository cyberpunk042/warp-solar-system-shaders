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

Stage 1 of the wisp's story: coast (the trap always returns you), burn (the drive
climbs, the map compresses), fall back (the fuel wall wins). Stage 2 — growing into
a body — and stage 3 — navigation — come later. See
``docs/research/57-the-wisp-in-the-box.md``.
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
