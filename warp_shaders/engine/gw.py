"""The gravitational-wave dictionary (host-side): the chirp, exactly.

Einstein 1916/1918 (quadrupole formula), Peters & Mathews 1963, Peters 1964 — the
closed-form core of binary inspiral, the physics LIGO turned into sound (G = c = 1):

* ``chirp_mass`` — the ONE mass combination the waveform measures:

      M_c = (m₁m₂)^{3/5} / (m₁+m₂)^{1/5}.

  Amplitude and frequency evolution both depend on the masses only through M_c at
  leading order; GW150914's M_c ≈ 30 M_sun was read straight off the chirp.
* ``gw_frequency`` — the quadrupole nature of gravity makes the wave oscillate at
  TWICE the orbital frequency: ``f_gw = 2 f_orb`` (the mass distribution repeats
  every half orbit). Asserted against ``orbital_frequency`` (Kepler: Ω² = M/a³).
* ``peters_dadt`` / ``peters_dedt`` — Peters 1964, exact at leading (2.5PN) order:

      da/dt = −(64/5)·m₁m₂M/a³ · (1−e²)^{−7/2}·(1 + 73e²/24 + 37e⁴/96)
      de/dt = −(304/15)·e·m₁m₂M/a⁴ · (1−e²)^{−5/2}·(1 + 121e²/304)

  Radiation reaction shrinks AND circularizes the orbit — integrate the coupled
  ODEs and eccentricity dies before the merger (asserted): that is why LIGO's
  binaries arrive circular.
* ``peters_merger_time`` — the circular-orbit lifetime

      T = 5a⁴/(256·m₁m₂·M):

  the FOURTH power. Halve the separation, SIXTEENFOLD less time left (asserted).
  Hulse-Taylor's pulsar (a ~ solar radii) has ~300 Myr; GW150914's last audible
  second covered the final ~350 km.
* ``chirp_frequency`` — invert T(a) with Kepler and the frequency runs away as

      f_gw(t) = (1/π)·(5/256)^{3/8} · M_c^{−5/8} · (t_c − t)^{−3/8}

  — the −3/8 power law IS the chirp (exponent asserted numerically); rising pitch
  and rising amplitude ``h ∝ M_c^{5/3} f^{2/3}/D`` (``strain_amplitude``) until
  coalescence at t_c.

See ``docs/research/53-gravitational-waves.md``.
"""

import math


def chirp_mass(m1: float, m2: float) -> float:
    """The chirp mass ``M_c = (m1·m2)^{3/5}/(m1+m2)^{1/5}`` (host-side): the single
    mass combination that sets the leading-order amplitude AND frequency evolution —
    what the waveform actually measures."""
    return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2


def orbital_frequency(a: float, m_total: float) -> float:
    """Kepler (host-side): ``f_orb = sqrt(M/a³)/2π``."""
    return math.sqrt(m_total / a ** 3) / (2.0 * math.pi)


def gw_frequency(a: float, m_total: float) -> float:
    """The wave frequency ``f_gw = 2·f_orb`` (host-side): gravity radiates through the
    quadrupole, and the binary's mass distribution repeats every HALF orbit — the
    factor of two asserted against Kepler."""
    return 2.0 * orbital_frequency(a, m_total)


def peters_dadt(a: float, e: float, m1: float, m2: float) -> float:
    """Peters 1964 semi-major-axis decay (host-side, exact leading order):
    ``da/dt = −64/5·m1·m2·M/a³·(1−e²)^{−7/2}(1 + 73e²/24 + 37e⁴/96)`` — always
    negative: radiation reaction only ever shrinks the orbit."""
    m_tot = m1 + m2
    enh = (1.0 + 73.0 * e * e / 24.0 + 37.0 * e ** 4 / 96.0) / (1.0 - e * e) ** 3.5
    return -(64.0 / 5.0) * m1 * m2 * m_tot / a ** 3 * enh


def peters_dedt(a: float, e: float, m1: float, m2: float) -> float:
    """Peters 1964 eccentricity decay (host-side): ``de/dt = −304/15·e·m1·m2·M/a⁴·
    (1−e²)^{−5/2}(1 + 121e²/304)`` — gravitational waves CIRCULARIZE: e only ever
    decreases (asserted by integrating the coupled system)."""
    m_tot = m1 + m2
    enh = (1.0 + 121.0 * e * e / 304.0) / (1.0 - e * e) ** 2.5
    return -(304.0 / 15.0) * e * m1 * m2 * m_tot / a ** 4 * enh


def peters_merger_time(a0: float, m1: float, m2: float) -> float:
    """Circular-orbit time to coalescence (host-side): ``T = 5a⁴/(256·m1·m2·M)`` —
    the fourth power: halve the separation, sixteenfold less time left (asserted)."""
    return 5.0 * a0 ** 4 / (256.0 * m1 * m2 * (m1 + m2))


def separation_of_time_left(t_left: float, m1: float, m2: float) -> float:
    """Invert the merger time (host-side): the separation when ``t_left`` remains,
    ``a = (256/5·m1·m2·M·t_left)^{1/4}`` — the inspiral trajectory a(t) in one line."""
    return (256.0 / 5.0 * m1 * m2 * (m1 + m2) * t_left) ** 0.25


def chirp_frequency(t_left: float, m_chirp: float) -> float:
    """THE CHIRP (host-side): ``f_gw(t) = (5/256)^{3/8}/(π·M_c^{5/8}·(t_c−t)^{3/8})``
    — frequency running away as the −3/8 power of time-to-merger (exponent asserted
    numerically). This curve, traced in the data, is how LIGO weighs black holes."""
    return (5.0 / 256.0) ** 0.375 / (math.pi * m_chirp ** 0.625 * t_left ** 0.375)


def strain_amplitude(f_gw: float, m_chirp: float, distance: float) -> float:
    """Leading-order strain (host-side): ``h ∝ M_c^{5/3}·(πf)^{2/3}/D`` — the
    amplitude also rises as the pitch rises: the chirp gets LOUDER as it climbs."""
    return 4.0 * m_chirp ** (5.0 / 3.0) * (math.pi * f_gw) ** (2.0 / 3.0) / distance


def evolve_peters(a0: float, e0: float, m1: float, m2: float, dt: float, n_steps: int):
    """Integrate the coupled Peters system (host-side, RK2): returns the (a, e)
    trajectory — the exact circularization curve the ``gw_orbits`` scene plays and
    the suite asserts (e strictly decreasing, dying faster than a)."""
    traj = [(a0, e0)]
    a, e = a0, e0
    for _ in range(n_steps):
        ka = peters_dadt(a, e, m1, m2)
        ke = peters_dedt(a, e, m1, m2)
        am = a + 0.5 * dt * ka
        em = max(e + 0.5 * dt * ke, 0.0)
        if am <= 0.05:
            break
        a = a + dt * peters_dadt(am, em, m1, m2)
        e = max(e + dt * peters_dedt(am, em, m1, m2), 0.0)
        if a <= 0.05:
            break
        traj.append((a, e))
    return traj
