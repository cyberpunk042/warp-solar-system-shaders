"""The ΛCDM dictionary (host-side): the expanding universe, exactly.

Friedmann 1922 wrote the equation; Hubble 1929 saw the recession; Riess and
Perlmutter 1998 measured distant supernovae TOO DIM for a decelerating universe —
the expansion is speeding up, and the 2011 Nobel went to the discovery that ~69% of
the universe is a cosmological constant. For the flat matter+Λ universe we actually
live in (late times; radiation negligible), the Friedmann equation has a CLOSED-FORM
solution — every law below is exact and test-asserted, with the real Planck 2018
numbers (H₀ = 67.36 km/s/Mpc, Ω_m = 0.3153):

* ``scale_factor`` — the exact solution of ``(ȧ/a)² = H₀²(Ω_m/a³ + Ω_Λ)``:

      a(t) = (Ω_m/Ω_Λ)^{1/3} · sinh^{2/3}( (3/2)·√Ω_Λ·H₀·t )

  matter-era ``t^{2/3}`` at early times, de Sitter exponential at late times, one
  smooth curve between (ASSERTED to satisfy the Friedmann ODE numerically).
* ``age_of_universe`` — invert at a = 1:

      t₀ = (2/(3√Ω_Λ H₀))·asinh(√(Ω_Λ/Ω_m)) = 13.8 Gyr   (asserted to 1%)

  — the age of everything, from two measured numbers and one exact formula.
* ``acceleration_onset`` — ä = 0 exactly at ``a = (Ω_m/2Ω_Λ)^{1/3}``, i.e.
  z ≈ 0.63 (asserted, incl. the numeric sign flip of ä): more distant supernovae
  sample the DECELERATING era; nearer ones the accelerating — the crossover IS the
  1998 signature.
* ``matter_lambda_equality`` — ρ_m = ρ_Λ at ``a = (Ω_m/Ω_Λ)^{1/3}``, z ≈ 0.3
  (asserted): we live just after the handoff.
* ``comoving_distance`` / ``luminosity_distance`` / ``distance_modulus`` — the
  Hubble-diagram machinery: ``D_C = (c/H₀)∫dz/E(z)``, ``D_L = (1+z)·D_C``,
  ``μ = 5·log₁₀(D_L/10pc)`` — the exact ΛCDM curve sits ABOVE the matter-only
  curve (dimmer supernovae, asserted): that gap is the Nobel.
* ``particle_horizon`` — how far light has EVER been able to come: ≈ 46 Gly
  comoving today (asserted; the neglected radiation era adds ~1 Gly more) — the
  radius of the observable universe, 3.3× ``c·t₀`` because space stretched behind
  the photons.
* ``event_horizon`` — how far light can EVER reach from here on: ≈ 16.7 Gly
  (asserted finite — a pure consequence of Λ): galaxies beyond it are sending
  light that will never arrive; the sky is slowly emptying.
* ``lookback_time`` — how long ago the light left a redshift-z source.

See ``docs/research/56-expanding-universe.md``.
"""

import math

# ---- Planck 2018 (TT,TE,EE+lowE+lensing) ----
H0_KM_S_MPC = 67.36
OMEGA_M = 0.3153
OMEGA_L = 1.0 - OMEGA_M          # flat
C_KM_S = 299792.458
MPC_KM = 3.0856775814913673e19
GYR_S = 3.1556952e16             # Julian Gyr in seconds
GLY_KM = C_KM_S * GYR_S          # one Gly in km


def hubble_h0_per_gyr() -> float:
    """H₀ converted to 1/Gyr (host-side): ``67.36 km/s/Mpc ≈ 0.0689/Gyr`` — the
    natural unit for the closed-form solution."""
    return H0_KM_S_MPC / MPC_KM * GYR_S


def e_of_z(z: float) -> float:
    """The dimensionless expansion rate (host-side): ``E(z) = H/H₀ =
    √(Ω_m(1+z)³ + Ω_Λ)`` — flat matter+Λ; every distance below integrates 1/E."""
    zp = 1.0 + z
    return math.sqrt(OMEGA_M * zp * zp * zp + OMEGA_L)


def scale_factor(t_gyr: float) -> float:
    """The EXACT flat-ΛCDM scale factor (host-side):
    ``a(t) = (Ω_m/Ω_Λ)^{1/3} sinh^{2/3}(1.5·√Ω_Λ·H₀·t)`` — asserted to satisfy
    the Friedmann equation numerically. a(t₀) = 1 (asserted)."""
    h0 = hubble_h0_per_gyr()
    x = 1.5 * math.sqrt(OMEGA_L) * h0 * t_gyr
    return (OMEGA_M / OMEGA_L) ** (1.0 / 3.0) * math.sinh(x) ** (2.0 / 3.0)


def age_of_universe() -> float:
    """The age of everything (host-side): invert a = 1,
    ``t₀ = (2/(3√Ω_Λ H₀))·asinh(√(Ω_Λ/Ω_m))`` ≈ 13.8 Gyr (asserted to 1%)."""
    h0 = hubble_h0_per_gyr()
    return 2.0 / (3.0 * math.sqrt(OMEGA_L) * h0) * math.asinh(math.sqrt(OMEGA_L / OMEGA_M))


def age_at_scale_factor(a: float) -> float:
    """When the universe was a times its present size (host-side): the closed-form
    inverse ``t(a) = (2/(3√Ω_Λ H₀))·asinh(√(Ω_Λ/Ω_m)·a^{3/2})``."""
    h0 = hubble_h0_per_gyr()
    return 2.0 / (3.0 * math.sqrt(OMEGA_L) * h0) * \
        math.asinh(math.sqrt(OMEGA_L / OMEGA_M) * a ** 1.5)


def acceleration_onset() -> float:
    """Where the brake became a throttle (host-side): ``ä = 0`` exactly at
    ``a = (Ω_m/(2Ω_Λ))^{1/3}`` — z ≈ 0.63 (asserted, plus the numeric sign flip):
    supernovae beyond it sample the decelerating universe — the 1998 signature."""
    return (OMEGA_M / (2.0 * OMEGA_L)) ** (1.0 / 3.0)


def matter_lambda_equality() -> float:
    """The handoff (host-side): ``ρ_m = ρ_Λ`` at ``a = (Ω_m/Ω_Λ)^{1/3}`` — z ≈ 0.3
    (asserted). Everything before was matter's era; everything after is Λ's."""
    return (OMEGA_M / OMEGA_L) ** (1.0 / 3.0)


def comoving_distance_gly(z: float, n: int = 4000) -> float:
    """Comoving distance to redshift z (host-side): ``D_C = (c/H₀)·∫₀ᶻ dz'/E(z')``
    (Simpson) in Gly — the ruler the Hubble diagram is built on."""
    if z <= 0.0:
        return 0.0
    h = z / float(n)
    s = 1.0 / e_of_z(0.0) + 1.0 / e_of_z(z)
    for k in range(1, n):
        s += (4.0 if k % 2 == 1 else 2.0) / e_of_z(h * float(k))
    dc_mpc = (C_KM_S / H0_KM_S_MPC) * s * h / 3.0
    return dc_mpc * MPC_KM / GLY_KM


def luminosity_distance_gly(z: float) -> float:
    """Luminosity distance (host-side): ``D_L = (1+z)·D_C`` (flat) — supernovae
    fade with its square; the extra (1+z) is redshift plus time dilation."""
    return (1.0 + z) * comoving_distance_gly(z)


def distance_modulus(z: float) -> float:
    """The Hubble-diagram y-axis (host-side): ``μ = 5·log₁₀(D_L/10 pc)`` — what
    supernova surveys actually plot. The exact ΛCDM curve sits ABOVE matter-only
    (dimmer; asserted): that gap is the 1998 discovery."""
    dl_pc = luminosity_distance_gly(z) * GLY_KM / MPC_KM * 1.0e6
    return 5.0 * math.log10(dl_pc / 10.0)


def distance_modulus_matter_only(z: float, n: int = 4000) -> float:
    """The ghost curve (host-side): the same distance modulus in a flat MATTER-ONLY
    universe (Ω_m = 1, exact ``D_C = (2c/H₀)(1−1/√(1+z))``) — what 1998 expected
    to fit, and didn't."""
    dc_mpc = 2.0 * (C_KM_S / H0_KM_S_MPC) * (1.0 - 1.0 / math.sqrt(1.0 + z))
    dl_pc = (1.0 + z) * dc_mpc * 1.0e6
    return 5.0 * math.log10(dl_pc / 10.0)


def particle_horizon_gly(t_gyr: float | None = None, n: int = 6000) -> float:
    """How far light has EVER been able to come (host-side): comoving
    ``D_PH = c·∫₀ᵗ dt'/a(t')`` — ≈ 46 Gly today (asserted): the radius of the
    observable universe, 3.3× naive c·t₀, because space stretched behind the light."""
    t0 = age_of_universe() if t_gyr is None else t_gyr
    h = t0 / float(n)
    s = 0.0
    for k in range(n):
        tm = h * (float(k) + 0.5)
        s += 1.0 / scale_factor(tm)
    return s * h  # in Gly: c = 1 Gly/Gyr


def event_horizon_gly(t_gyr: float | None = None, n: int = 6000,
                      t_max_gyr: float = 500.0) -> float:
    """How far light can EVER reach from here on (host-side): comoving
    ``D_EH = c·∫ₜ^∞ dt'/a(t')`` — FINITE (≈ 16.7 Gly today, asserted) purely
    because of Λ: galaxies beyond it are already sending light that will never
    arrive. The sky is slowly emptying."""
    t0 = age_of_universe() if t_gyr is None else t_gyr
    h = (t_max_gyr - t0) / float(n)
    s = 0.0
    for k in range(n):
        tm = t0 + h * (float(k) + 0.5)
        s += 1.0 / scale_factor(tm)
    return s * h


def lookback_time(z: float) -> float:
    """How long ago the light left (host-side): ``t₀ − t(a = 1/(1+z))`` in Gyr —
    the Hubble diagram's hidden time axis."""
    return age_of_universe() - age_at_scale_factor(1.0 / (1.0 + z))
