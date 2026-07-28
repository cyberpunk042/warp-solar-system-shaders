"""The Kerr dictionary (host-side): rotation, the ergosphere, the mine, the bomb.

Everything a spinning black hole does that a static one cannot, in exact closed forms
(G = c = 1, spin 0 ≤ a ≤ M):

* ``kerr_horizons`` — ``r_± = M ± √(M² − a²)``, with the exact identities
  ``r_+ + r_− = 2M`` and ``r_+·r_− = a²`` (test-asserted). At a = M the horizons merge:
  the extremal hole.
* ``ergosurface`` — ``r_E(θ) = M + √(M² − a²cos²θ)``: touches the horizon at the poles,
  bulges to ``2M`` at the equator. Between r_+ and r_E lies the ERGOREGION, where
  standing still is impossible: spacetime itself rotates.
* ``kerr_omega_h`` — the horizon's angular velocity ``Ω_H = a/(2Mr_+)`` (the identity
  ``r_+² + a² = 2Mr_+`` makes the two standard forms equal).
* ``kerr_temperature`` — ``T = √(M² − a²)/(4πMr_+)``: reduces to Schwarzschild's
  ``1/8πM`` at a = 0 and → 0 at extremality — the third law: you cannot spin a hole to
  a = M in finite steps.
* ``kerr_entropy`` — ``S = A/4 = 2πMr_+``.
* ``irreducible_mass`` — Christodoulou 1970: ``M_irr = √(Mr_+/2)``, the part of the
  mass you can NEVER extract (``M_irr² ∝ area``). The rotational store
  ``M − M_irr`` reaches its maximum ``(1 − 1/√2)M ≈ 29.3%`` at extremality — the
  Penrose bound (``penrose_bound``).
* ``penrose_extract`` — one Penrose-process event: a particle splits in the ergoregion,
  the negative-energy fragment falls in, energy δE comes out. Parameterized by
  reversibility q ∈ (0,1]: ``δJ = −δE/(q·Ω_H)``; the first law then gives
  ``dA ∝ dM − Ω_H·dJ = −δE(1 − 1/q) ≥ 0`` — **the area theorem is built into the
  bookkeeping and asserted over random extraction sequences**: M and J fall, the
  horizon AREA (and M_irr) never does.
* ``superradiant`` — Zel'dovich/Misner: a wave of frequency ω and azimuthal number m
  scatters off the hole AMPLIFIED iff ``0 < ω < m·Ω_H`` (the wave analogue of Penrose).
* ``bomb_amplitude`` — Press-Teukolsky 1972: wrap the hole in a mirror and the
  amplified wave re-amplifies — ``A_n = A₀(1 + g)ⁿ``, exponential runaway: the
  black-hole bomb.
* ``lense_thirring_omega`` — the frame-drag rate ``ω_LT = 2Ma/r³`` (exact
  asymptotically): the rate at which local inertial frames are dragged around the spin
  axis — what the ray tracer applies along photon paths.

See ``docs/research/50-kerr-spinning-black-hole.md``.
"""

import math


def kerr_horizons(m: float, a: float):
    """Outer and inner horizons ``r_± = M ± √(M² − a²)`` (host-side). Requires a ≤ M
    (cosmic censorship — a > M would be a naked singularity)."""
    d = math.sqrt(max(m * m - a * a, 0.0))
    return (m + d, m - d)


def ergosurface(m: float, a: float, theta: float) -> float:
    """The stationary limit ``r_E(θ) = M + √(M² − a²cos²θ)`` (host-side): inside it no
    observer can stand still — the ergoregion, where the Penrose process lives."""
    c = math.cos(theta)
    return m + math.sqrt(max(m * m - a * a * c * c, 0.0))


def kerr_omega_h(m: float, a: float) -> float:
    """Angular velocity of the horizon ``Ω_H = a/(2Mr_+)`` (host-side): the rate at
    which the horizon — and everything close enough to it — is forced to rotate."""
    r_p, _ = kerr_horizons(m, a)
    return a / (2.0 * m * r_p)


def kerr_temperature(m: float, a: float) -> float:
    """Hawking temperature ``T = κ/2π = √(M² − a²)/(4πMr_+)`` (host-side): the
    Schwarzschild ``1/8πM`` at a = 0, and → 0 at extremality (the third law)."""
    r_p, _ = kerr_horizons(m, a)
    return math.sqrt(max(m * m - a * a, 0.0)) / (4.0 * math.pi * m * r_p)


def kerr_entropy(m: float, a: float) -> float:
    """Bekenstein-Hawking entropy ``S = A/4 = 2πMr_+`` (host-side); the horizon area is
    ``A = 4π(r_+² + a²) = 8πMr_+``."""
    r_p, _ = kerr_horizons(m, a)
    return 2.0 * math.pi * m * r_p


def irreducible_mass(m: float, a: float) -> float:
    """Christodoulou's irreducible mass ``M_irr = √(Mr_+/2)`` (host-side): the
    area-locked part of the mass. M_irr² is proportional to the horizon area, so the
    area theorem reads: M_irr NEVER decreases. Rotational energy ``M − M_irr`` is the
    mine's total store."""
    r_p, _ = kerr_horizons(m, a)
    return math.sqrt(0.5 * m * r_p)


def penrose_bound() -> float:
    """The maximum extractable fraction of an extremal hole's mass (host-side):
    ``1 − M_irr/M = 1 − 1/√2 ≈ 0.2929`` — the Penrose bound."""
    return 1.0 - 1.0 / math.sqrt(2.0)


def penrose_extract(m: float, a: float, de: float, q: float = 0.9):
    """One Penrose-process event (host-side): extract energy ``de``, spinning the hole
    down by ``δJ = de/(q·Ω_H)`` with reversibility q ∈ (0,1] (q = 1: reversible,
    area-preserving; q < 1: irreversible, area grows). Clamps ``de`` so the spin never
    goes negative. Returns ``(m', a', de_actual)``."""
    if a <= 1e-12:
        return (m, a, 0.0)                    # no ergoregion left: the mine is exhausted
    # integrate in substeps so Omega_H is fresh along the extraction — a single finite
    # step with the initial Omega_H can spuriously violate dA >= 0
    done = 0.0
    remaining = de
    while remaining > 1e-15 and a > 1e-12:
        step = min(remaining, 2.0e-4)
        om = kerr_omega_h(m, a)
        j = a * m
        step = min(step, q * om * j)          # can't extract past a' = 0
        if step <= 0.0:
            break
        m -= step
        j -= step / (q * om)
        a = min(max(j / m, 0.0), m)
        done += step
        remaining -= step
    return (m, a, done)


def superradiant(omega: float, m_azim: int, omega_h: float) -> bool:
    """The superradiance condition (host-side): a wave mode e^{−iωt + imφ} scatters off
    the horizon AMPLIFIED iff ``0 < ω < m·Ω_H`` (Zel'dovich 1971; Misner). The wave
    analogue of the Penrose process — the reflected wave carries away rotational
    energy."""
    return 0.0 < omega < m_azim * omega_h


def bomb_amplitude(n: int, gain: float, a0: float = 1.0) -> float:
    """The black-hole bomb (host-side, Press-Teukolsky 1972): enclose a superradiant
    hole in a mirror and each round trip multiplies the wave by (1 + gain):
    ``A_n = A₀·(1 + g)ⁿ`` — exponential runaway until the mirror bursts or the spin is
    exhausted."""
    return a0 * (1.0 + gain) ** n


def lense_thirring_omega(r: float, m: float, a: float) -> float:
    """Frame-dragging angular velocity ``ω_LT = 2Ma/r³`` (host-side; exact in the
    asymptotic regime): the rate at which local inertial frames — and light — are
    dragged around the spin axis. Falls as 1/r³: doubling r divides the drag by 8."""
    return 2.0 * m * a / (r * r * r)
