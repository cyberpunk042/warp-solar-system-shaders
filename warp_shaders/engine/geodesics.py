"""The Schwarzschild geodesic dictionary (host-side): the classic tests, exactly.

Einstein 1915 computed Mercury's missing 43″/century three days before the field
equations were final; Shapiro 1964 predicted radar echoes past the Sun would return
late; Pound-Rebka 1959 weighed a photon climbing 22.5 m of Harvard tower; and every
GPS fix silently corrects +38 μs/day of relativistic clock drift. All of it lives in
ONE metric — Schwarzschild — and all of it is closed-form and test-asserted here
(geometric units G = c = 1; masses are lengths; real-unit constants at the bottom):

* ``veff_sq`` — the timelike effective potential ``V²(r) = (1−2M/r)(1+L²/r²)``: the
  whole solar system is a bead sliding on this one curve.
* ``circular_angular_momentum`` / ``circular_energy`` — circular orbits sit at
  ``V′ = 0`` exactly: ``L² = Mr²/(r−3M)``, ``E = (1−2M/r)/√(1−3M/r)`` (asserted).
* ``isco_radius`` — stability runs out at ``r = 6M`` (``V″ = 0``), where
  ``E = √(8/9)`` and ``L = 2√3·M`` (asserted): the inner edge of accretion disks.
* ``photon_sphere`` / ``critical_impact_parameter`` — light circles at ``r = 3M``;
  rays inside ``b_c = 3√3·M`` fall in — the shadow the EHT photographed.
* ``integrate_orbit`` — the EXACT bound geodesic, RK4 on the Binet equation

      d²u/dφ² + u = M/L² + 3Mu²,   u = 1/r

  (the ``3Mu²`` term IS general relativity — drop it and Kepler's closed ellipse
  returns). ``measured_precession`` reads the apsidal advance off the integration;
  asserted against the closed form below in the weak field, and asserted to EXCEED
  it in the strong field.
* ``precession_per_orbit`` — Einstein 1915: ``Δφ = 6πM/(a(1−e²))`` per orbit.
  ``mercury_precession_arcsec_century()`` feeds in the real Mercury and returns
  ≈ 42.98″/century (asserted) — the number that convinced Einstein himself.
* ``deflection_angle`` — ``α = 4M/b``: Eddington's 1919 eclipse, 1.75″ at the solar
  limb (asserted; the weak-field anchor of the round-12 lensing set).
* ``shapiro_roundtrip_excess`` — radar to a planet at superior conjunction returns
  late by ``Δt = 4M·[ln(4·r₁r₂/b²) + 1]``: light signals near the Sun travel with
  coordinate speed < 1. Earth-Mars grazing the limb: ≈ 250 μs (asserted).
* ``clock_rate_static`` / ``clock_rate_orbit`` — ``√(1−2M/r)`` standing still,
  ``√(1−3M/r)`` on a circular orbit (gravity + orbital time dilation in one term).
* ``clock_crossover_radius`` — an orbiting clock ticks at exactly the ground rate at
  ``r = 3R/2``, INDEPENDENT of the mass (1−3M/r = 1−2M/R ⇒ r = 3R/2, asserted):
  below it (ISS) clocks run slow, above it (GPS) they run fast.
* ``gps_daily_drift_us`` — the real number: +38.5 μs/day (asserted), corrected in
  every receiver on Earth. ``pound_rebka_shift`` — gh/c² ≈ 2.46×10⁻¹⁵ over 22.5 m
  (asserted): the first terrestrial weighing of gravitational time dilation.

See ``docs/research/55-classic-tests.md``.
"""

import math

# ---- real-unit constants (km; masses are GM/c² in km) ----
M_SUN_KM = 1.4766250385          # the Sun's half-Schwarzschild radius GM/c²
R_SUN_KM = 6.957e5               # the solar limb — Eddington's & Shapiro's b
M_EARTH_KM = 4.4347e-6           # GM_earth/c² in km
R_EARTH_KM = 6371.0
R_GPS_KM = 26561.75              # GPS orbit radius (semi-major axis)
AU_KM = 1.495978707e8
C_KM_S = 299792.458

MERCURY_A_KM = 5.7909050e7       # Mercury: semi-major axis, eccentricity, period
MERCURY_E = 0.205630
MERCURY_PERIOD_DAYS = 87.9691


def veff_sq(r: float, ell: float, mass: float) -> float:
    """The timelike effective potential squared (host-side):
    ``V²(r) = (1−2M/r)(1+L²/r²)`` — turning points solve ``E² = V²``; circular
    orbits sit at its stationary points (asserted via the closed forms below)."""
    return (1.0 - 2.0 * mass / r) * (1.0 + ell * ell / (r * r))


def circular_angular_momentum(r: float, mass: float) -> float:
    """Angular momentum of the circular orbit at radius r (host-side):
    ``L = r·√(M/(r−3M))`` — exact solution of V′ = 0; diverges at the photon
    sphere r = 3M where no massive particle can keep up."""
    return r * math.sqrt(mass / (r - 3.0 * mass))


def circular_energy(r: float, mass: float) -> float:
    """Energy per unit mass of the circular orbit (host-side):
    ``E = (1−2M/r)/√(1−3M/r)`` — E < 1 is bound; at the ISCO E = √(8/9), so 5.7%
    of rest mass has been radiated away: accretion power (asserted)."""
    return (1.0 - 2.0 * mass / r) / math.sqrt(1.0 - 3.0 * mass / r)


def isco_radius(mass: float) -> float:
    """The innermost stable circular orbit (host-side): ``r = 6M`` exactly
    (V″ = 0) — inside it, no amount of angular momentum stabilizes the orbit;
    the inner edge of thin accretion disks."""
    return 6.0 * mass


def photon_sphere(mass: float) -> float:
    """Light's circular orbit (host-side): ``r = 3M`` exactly — the unstable rim
    the EHT ring traces."""
    return 3.0 * mass


def critical_impact_parameter(mass: float) -> float:
    """The capture threshold for light (host-side): ``b_c = 3√3·M ≈ 5.196M`` —
    rays aimed inside it spiral in; the black-hole shadow's radius."""
    return 3.0 * math.sqrt(3.0) * mass


def precession_per_orbit(a: float, e: float, mass: float) -> float:
    """Einstein 1915 (host-side): perihelion advance per orbit
    ``Δφ = 6πM/(a(1−e²))`` radians — leading order; the exact integrated orbit
    exceeds it in the strong field (asserted)."""
    return 6.0 * math.pi * mass / (a * (1.0 - e * e))


def integrate_orbit(a: float, e: float, mass: float, n_steps: int = 20000,
                    phi_max: float = 4.0 * math.pi):
    """Integrate the EXACT bound geodesic (host-side): RK4 on the Binet equation
    ``u″ + u = M/L² + 3Mu²`` from perihelion (u = 1/(a(1−e)), u′ = 0), with the
    Newtonian mapping ``L² = Ma(1−e²)``. Returns (phi_list, u_list) sampled
    uniformly in φ — the rosette the ``gr_precession`` scene draws, and the
    trajectory ``measured_precession`` reads the apsidal advance from."""
    ell_sq = mass * a * (1.0 - e * e)
    c0 = mass / ell_sq
    u = 1.0 / (a * (1.0 - e))
    v = 0.0                       # du/dphi
    h = phi_max / float(n_steps)
    phis = [0.0]
    us = [u]

    def acc(uu):
        return c0 + 3.0 * mass * uu * uu - uu

    for k in range(n_steps):
        k1u, k1v = v, acc(u)
        k2u, k2v = v + 0.5 * h * k1v, acc(u + 0.5 * h * k1u)
        k3u, k3v = v + 0.5 * h * k2v, acc(u + 0.5 * h * k2u)
        k4u, k4v = v + h * k3v, acc(u + h * k3u)
        u = u + (h / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        v = v + (h / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        phis.append(h * float(k + 1))
        us.append(u)
    return phis, us


def measured_precession(a: float, e: float, mass: float,
                        n_steps: int = 60000) -> float:
    """Read the apsidal advance off the exact integration (host-side): find the
    first return to perihelion (the next maximum of u) and return ``Δφ − 2π``.
    Asserted ≈ the closed form in the weak field and > it in the strong field."""
    phis, us = integrate_orbit(a, e, mass, n_steps, 2.0 * math.pi * 1.6)
    # perihelion = local max of u after leaving the initial one
    for k in range(2, len(us) - 1):
        if us[k] >= us[k - 1] and us[k] >= us[k + 1] and phis[k] > math.pi:
            # quadratic interpolation of the extremum for sub-step precision
            y0, y1, y2 = us[k - 1], us[k], us[k + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 0.0 else 0.0
            h = phis[1] - phis[0]
            return phis[k] + delta * h - 2.0 * math.pi
    raise RuntimeError("no perihelion return found — increase phi_max")


def mercury_precession_arcsec_century() -> float:
    """The number that started it all (host-side): Mercury's real elements fed
    through ``6πM/(a(1−e²))`` × orbits/century → ≈ 42.98″ (asserted) — the
    anomaly Le Verrier logged in 1859 and Einstein recovered in 1915."""
    dphi = precession_per_orbit(MERCURY_A_KM, MERCURY_E, M_SUN_KM)
    orbits = 36525.0 / MERCURY_PERIOD_DAYS
    return dphi * orbits * (180.0 / math.pi) * 3600.0


def deflection_angle(b: float, mass: float) -> float:
    """Weak-field light bending (host-side): ``α = 4M/b`` — 1.75″ at the solar
    limb (asserted): Eddington 1919, and the anchor the round-12 lensing set
    builds on."""
    return 4.0 * mass / b


def shapiro_roundtrip_excess(r1: float, r2: float, b: float, mass: float) -> float:
    """The fourth test (host-side): round-trip radar excess past a mass,
    ``Δ(ct) = 4M·[ln(4·r₁r₂/b²) + 1]`` (returned as a LENGTH; divide by c for
    seconds) — light near the Sun travels with coordinate speed 1−2M/r < 1.
    Earth-Mars at the solar limb ≈ 250 μs (asserted)."""
    return 4.0 * mass * (math.log(4.0 * r1 * r2 / (b * b)) + 1.0)


def coordinate_light_speed(r: float, mass: float) -> float:
    """Radial coordinate speed of light (host-side): ``c(r) = 1 − 2M/r`` (isotropic
    leading order) — the slowing the ``gr_shapiro`` scene animates through the
    solar corona; integrated along the path it IS the Shapiro delay."""
    return 1.0 - 2.0 * mass / r


def clock_rate_static(r: float, mass: float) -> float:
    """Proper-time rate of a clock standing at r (host-side): ``dτ/dt = √(1−2M/r)``
    — deeper is slower: Pound-Rebka weighed this over 22.5 m of tower."""
    return math.sqrt(1.0 - 2.0 * mass / r)


def clock_rate_orbit(r: float, mass: float) -> float:
    """Proper-time rate of a clock on a CIRCULAR ORBIT at r (host-side):
    ``dτ/dt = √(1−3M/r)`` — gravitational blueshift with altitude plus orbital
    time dilation, folded into one exact term."""
    return math.sqrt(1.0 - 3.0 * mass / r)


def clock_crossover_radius(r_surface: float) -> float:
    """The break-even altitude (host-side): an orbiting clock ticks at exactly the
    surface rate when ``1−3M/r = 1−2M/R`` ⇒ ``r = 3R/2`` — INDEPENDENT of the
    mass (asserted). Below it (ISS) astronauts age slower than Earth; above it
    (GPS) clocks run fast."""
    return 1.5 * r_surface


def gps_daily_drift_us(r_orbit: float = R_GPS_KM) -> float:
    """The engineering test (host-side): a GPS clock's drift against the ground,
    ``(√(1−3M/r)/√(1−2M/R) − 1)·86400 s`` in μs/day → ≈ +38.5 (asserted) —
    uncorrected, GPS fixes would wander ~11 km/day."""
    rate = clock_rate_orbit(r_orbit, M_EARTH_KM) / clock_rate_static(R_EARTH_KM, M_EARTH_KM)
    return (rate - 1.0) * 86400.0 * 1.0e6


def pound_rebka_shift(height_km: float = 22.5e-3) -> float:
    """Pound-Rebka 1959 (host-side): fractional blueshift of a photon falling
    h = 22.5 m at Harvard, ``z = M·h/R²`` (weak field) ≈ 2.46×10⁻¹⁵ (asserted)
    — resolved with the Mössbauer effect; gravity weighed in a stairwell."""
    return M_EARTH_KM * height_km / (R_EARTH_KM * R_EARTH_KM)
