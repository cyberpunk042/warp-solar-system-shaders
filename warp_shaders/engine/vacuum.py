"""The quantum-vacuum dictionary (host-side): the vacuum is alive.

Three exact results proving empty space is a physical medium — and completing the
temperature trilogy the horizon arc built (ħ = c = 1 throughout):

* ``unruh_temperature`` — Unruh 1976: an observer with proper acceleration ``a``
  measures the Minkowski vacuum as a thermal bath at

      T = a/2π.

  This is the third member of one identity. Hawking: ``T = κ/2π`` (κ = surface
  gravity). Gibbons-Hawking: ``T = H/2π``. Unruh: ``T = a/2π``. All three are the
  SAME statement — a horizon plus quantum fields equals heat — and the suite asserts
  the trilogy numerically (feed κ, H, a with equal values: equal temperatures).
* ``rindler_horizon_distance`` — the accelerating observer's horizon trails at fixed
  proper distance ``d = 1/a`` behind them: accelerate harder and your own private
  horizon closes in. Their worldline is the hyperbola ``x² − t² = 1/a²``
  (``rindler_worldline``), forever inside the right wedge; events behind the null
  line x = t are unseeable — a horizon with no black hole anywhere.
* ``casimir_pressure`` — Casimir 1948: two parallel mirrors a gap ``d`` apart exclude
  every vacuum mode that doesn't fit ``k_n = nπ/d``; the missing modes leave less
  outward push inside than outside, and the plates feel

      P = −π²/240 · 1/d⁴        (E/A = −π²/720 · 1/d³, ``casimir_energy``)

  attraction from *nothing*. The fourth-power law is brutal — halve the gap,
  SIXTEENFOLD the pressure (asserted exactly) — and real: measured to ~1% (Lamoreaux
  1997, Mohideen-Roy 1998). ``allowed_modes`` counts the surviving standing waves.
* ``schwinger_rate`` — Schwinger 1951 (after Sauter): a static electric field E tries
  to pull virtual e⁺e⁻ pairs apart; when the work over a Compton wavelength rivals
  2m, the pairs come REAL. The rate per volume

      Γ ∝ E² · exp(−π·E_c/E),    E_c = m²/e  (~1.3×10¹⁸ V/m)

  is non-perturbative — essentially exactly zero below E_c (at E = E_c/10 the
  exponential alone is e^{−10π} ≈ 10⁻¹⁴ of critical, asserted), then an avalanche:
  the vacuum breaks down like a dielectric and shorts the field out. Next-generation
  lasers (ELI) chase exactly this edge.

See ``docs/research/52-quantum-vacuum.md``.
"""

import math


def unruh_temperature(a: float) -> float:
    """The Unruh temperature ``T = a/2π`` (host-side): uniform proper acceleration a
    turns the Minkowski vacuum into a thermal bath. The flat-space member of the
    κ/2π — H/2π — a/2π trilogy."""
    return a / (2.0 * math.pi)


def rindler_horizon_distance(a: float) -> float:
    """Proper distance from a uniformly-accelerating observer to their Rindler horizon
    (host-side): ``d = 1/a``. Harder acceleration pulls your private horizon closer."""
    return 1.0 / a


def rindler_worldline(a: float, tau: float):
    """The accelerated worldline in inertial coordinates (host-side):
    ``x(τ) = cosh(aτ)/a, t(τ) = sinh(aτ)/a`` — the hyperbola x² − t² = 1/a², asymptotic
    to the horizon null line x = t, never crossing it."""
    return (math.cosh(a * tau) / a, math.sinh(a * tau) / a)


def casimir_pressure(d: float) -> float:
    """Casimir pressure between ideal parallel plates a gap d apart (host-side):
    ``P = −π²/(240 d⁴)`` — negative: the plates are pushed TOGETHER by the modes that
    don't fit between them."""
    return -math.pi ** 2 / (240.0 * d ** 4)


def casimir_energy(d: float) -> float:
    """Casimir energy per unit plate area (host-side): ``E/A = −π²/(720 d³)``;
    ``P = −∂(E/A)/∂d`` recovers the pressure (asserted)."""
    return -math.pi ** 2 / (720.0 * d ** 3)


def allowed_modes(d: float, k_max: float) -> int:
    """How many standing waves fit between the plates below a cutoff (host-side):
    ``k_n = nπ/d ≤ k_max`` → ``⌊k_max·d/π⌋``. Shrink the gap and modes are evicted —
    the eviction IS the force."""
    return int(k_max * d / math.pi)


def schwinger_critical_field(m: float = 1.0, e: float = 1.0) -> float:
    """The Schwinger critical field ``E_c = m²/e`` (host-side; ~1.3×10¹⁸ V/m for
    electrons): where the work done separating a virtual pair over its Compton
    wavelength reaches the pair's rest mass."""
    return m * m / e


def schwinger_rate(field: float, e_crit: float) -> float:
    """Pair-production rate per unit volume (leading Schwinger term, host-side):
    ``Γ ∝ E² exp(−π E_c/E)`` — non-perturbative, essentially zero below E_c, an
    avalanche above."""
    if field <= 0.0:
        return 0.0
    return field * field * math.exp(-math.pi * e_crit / field)
