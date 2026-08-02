"""The 2D Ising model, exactly (host-side): Onsager, Yang, Kramers-Wannier.

The square-lattice Ising ferromagnet (J = k_B = 1) is the crown jewel of
exactly-solvable statistical mechanics: an interacting, phase-transitioning
system whose critical point, energy, and order parameter are all CLOSED FORMS —
and every law below is test-asserted, several at machine precision:

* ``critical_temperature`` — Onsager (1944): ``T_c = 2/ln(1+√2) ≈ 2.269185``.
  The self-duality that pins it: ``sinh(2/T_c) = 1`` EXACTLY (asserted at
  machine precision).
* ``magnetization_exact`` — Yang (1952): below T_c the spontaneous
  magnetization is ``M = (1 − sinh(2/T)^{−4})^{1/8}``, and EXACTLY ZERO above.
  The 1/8 is the exact critical exponent β: the log-slope of M against
  (T_c − T) is asserted to equal 0.125 to five digits. No mean-field 1/2 —
  the exact answer.
* ``internal_energy_exact`` — Onsager's energy per spin,
  ``U = −coth(2K)[1 + (2/π)(2tanh²2K − 1)·K₁(κ)]`` with
  ``κ = 2 sinh 2K/cosh²2K`` and K₁ the complete elliptic integral (computed
  here by AGM). Asserted: ``U → −2`` as T → 0 (the ground state),
  ``U(T_c) = −√2`` EXACTLY (the elliptic term vanishes at criticality because
  ``2tanh²(2K_c) = 1``), and ``U ≈ −2/T`` at high T.
* ``dual_temperature`` — Kramers-Wannier (1941): the high-T and low-T phases
  are images of each other under ``sinh(2/T)·sinh(2/T*) = 1`` (product asserted
  = 1; involution T** = T asserted; fixed point T_c asserted). The critical
  point was KNOWN before the model was solved: it is the only temperature that
  maps to itself.
* ``metropolis_sweep`` — a deterministic (seeded) NumPy checkerboard Metropolis
  reference. The suite runs it and asserts the SIMULATION lands on Yang's
  closed form — measured, not asserted twice over: |M_sim(1.5) − 0.9865| small,
  M_sim(3.5) ≈ 0 — and the scenes run the same dynamics live on the Warp
  lattice against the exact curve.

Scenes: ``ising_quench`` (a live lattice cooled through T_c — paramagnetic
snow, critical clusters at every scale, coarsening domains),
``ising_magnetization`` (live lattices landing on Yang's curve),
``ising_duality`` (two lattices locked by Kramers-Wannier, meeting at the
self-dual point). See ``docs/research/58-ising-exactly.md``.
"""

import math

import numpy as np


def critical_temperature() -> float:
    """Onsager's exact critical point (host-side): ``T_c = 2/ln(1+√2)`` —
    pinned by self-duality: ``sinh(2/T_c) = 1`` EXACTLY (asserted)."""
    return 2.0 / math.log(1.0 + math.sqrt(2.0))


def magnetization_exact(t: float) -> float:
    """Yang's spontaneous magnetization (host-side):
    ``M = (1 − sinh(2/T)^{−4})^{1/8}`` below T_c, EXACTLY zero above.
    The exponent 1/8 is the exact β (log-slope asserted = 0.125)."""
    if t >= critical_temperature():
        return 0.0
    s = math.sinh(2.0 / t)
    return (1.0 - s ** -4) ** 0.125


def ellipk_agm(k: float) -> float:
    """Complete elliptic integral of the first kind K(k) by the
    arithmetic-geometric mean (host-side): ``K = π/(2·AGM(1, √(1−k²)))``.
    Asserted: K(0) = π/2."""
    a, b = 1.0, math.sqrt(max(1.0 - k * k, 1e-300))
    for _ in range(60):
        a, b = 0.5 * (a + b), math.sqrt(a * b)
    return math.pi / (2.0 * a)


def internal_energy_exact(t: float) -> float:
    """Onsager's energy per spin (host-side): asserted −2 at T → 0,
    EXACTLY −√2 at T_c (the elliptic term's coefficient ``2tanh²2K − 1``
    vanishes at criticality), and ≈ −2/T at high T."""
    kk = 1.0 / t
    kap = 2.0 * math.sinh(2.0 * kk) / math.cosh(2.0 * kk) ** 2
    kp = 2.0 * math.tanh(2.0 * kk) ** 2 - 1.0
    kap = min(kap, 1.0 - 1e-15)
    return -1.0 / math.tanh(2.0 * kk) * \
        (1.0 + (2.0 / math.pi) * kp * ellipk_agm(kap))


def dual_temperature(t: float) -> float:
    """Kramers-Wannier (host-side): the dual temperature with
    ``sinh(2/T)·sinh(2/T*) = 1`` (product asserted = 1 exactly; involution
    asserted; T_c is the unique fixed point, asserted). Order and disorder
    are the same model read twice."""
    return 2.0 / math.asinh(1.0 / math.sinh(2.0 / t))


def metropolis_sweep(spins: np.ndarray, t: float, rng: np.random.Generator) -> None:
    """One deterministic checkerboard Metropolis sweep IN PLACE (host-side
    NumPy reference; the scenes run the same dynamics as a Warp kernel).
    Periodic boundaries; J = 1. The suite asserts this simulation lands on
    Yang's exact curve — measured, not asserted twice over."""
    n = spins.shape[0]
    iy, ix = np.mgrid[0:n, 0:n]
    for parity in (0, 1):
        mask = ((ix + iy) % 2) == parity
        nb = (np.roll(spins, 1, 0) + np.roll(spins, -1, 0) +
              np.roll(spins, 1, 1) + np.roll(spins, -1, 1))
        d_e = 2.0 * spins * nb
        accept = (d_e <= 0.0) | (rng.random(spins.shape) < np.exp(-d_e / t))
        spins[mask & accept] *= -1


def simulate_magnetization(n: int, t: float, sweeps: int, seed: int = 7) -> float:
    """Run a seeded lattice to quasi-equilibrium and return |M| averaged over
    the last quarter of the run (host-side). Deterministic for fixed inputs —
    the suite compares it against ``magnetization_exact``."""
    rng = np.random.default_rng(seed)
    spins = np.where(rng.random((n, n)) < 0.5, 1.0, -1.0)
    tail = []
    for k in range(sweeps):
        metropolis_sweep(spins, t, rng)
        if k >= 3 * sweeps // 4:
            tail.append(abs(float(spins.mean())))
    return float(np.mean(tail))
