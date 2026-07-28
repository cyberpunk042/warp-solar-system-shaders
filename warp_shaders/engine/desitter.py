"""The de Sitter dictionary (host-side): the horizon that wraps around YOU.

A universe dominated by a cosmological constant expands as ``a(t) = e^{Ht}``, and every
observer is surrounded by a **cosmic event horizon** at proper radius ``1/H`` — light
emitted beyond it never arrives. It is a black-hole horizon turned inside out, and it
obeys the same thermodynamics (Gibbons & Hawking 1977), in closed forms (G = c = ħ = 1):

* ``horizon_radius`` — ``r_H = 1/H``: the edge of the observable, per observer.
* ``gibbons_hawking_temperature`` — ``T = H/2π``: empty space at constant Λ is a thermal
  bath. For our universe (H₀ ≈ 10⁻³³ eV) that is ~10⁻³⁰ K — real but unmeasurably cold.
* ``ds_entropy`` — ``S = A/4 = π/H²``: the largest entropy any horizon-bounded patch can
  have. For our Λ: ~10¹²² — the famous number, and the ultimate bound on what our
  universe can ever contain (Bousso).
* ``comoving_event_horizon`` — ``χ_EH(t) = e^{−Ht}/H``: in comoving coordinates the
  horizon SHRINKS exponentially — the mechanism behind inflation's magic trick.
* ``mode_crossing_time`` — a comoving wave k oscillates while inside the horizon
  (k > aH) and **freezes** when it exits (k = aH), at ``t_k = ln(k/H)/H``: crossing
  times are LOGARITHMIC in k (asserted) — equal multiplicative steps in scale leave
  the horizon at equal time steps. Frozen amplitudes become classical.
* ``mode_amplitude`` — the frozen curvature amplitude with a slow-roll tilt:
  ``P(k) ∝ k^{n_s−1}`` (``spectral_tilt``: ``n_s − 1 = 2η − 6ε``, Planck measures
  n_s ≈ 0.965 — slightly red, the fingerprint of the inflaton's slow roll). The
  power-law is asserted exactly.
* ``efolds`` — ``N = ln(a_end/a_start)``; solving the horizon/flatness problems needs
  N ≳ 60 (asserted arithmetic: e⁶⁰ ≈ 10²⁶ stretches a Planck patch beyond today's
  Hubble volume).

Inflation in one sentence, all of it computable here: for 60 e-folds the comoving
horizon shrinks, quantum modes exit and freeze with a nearly-flat, slightly-red
spectrum; when inflation ends the horizon grows back, the frozen modes **re-enter**,
and gravity amplifies them into galaxies — the CMB's ripples are microscopic vacuum
noise worn at cosmological size. See ``docs/research/51-desitter-cosmic-horizon.md``.
"""

import math


def horizon_radius(h: float) -> float:
    """Proper radius of the de Sitter event horizon, ``r_H = 1/H`` (host-side)."""
    return 1.0 / h


def gibbons_hawking_temperature(h: float) -> float:
    """The Gibbons-Hawking temperature ``T = H/2π`` (host-side): an inertial observer
    in de Sitter space detects thermal radiation from their own horizon."""
    return h / (2.0 * math.pi)


def ds_entropy(h: float) -> float:
    """Horizon entropy ``S = A/4 = π/H²`` (host-side; horizon area ``A = 4π/H²``) —
    the holographic bound on the observer's patch."""
    return math.pi / (h * h)


def comoving_event_horizon(h: float, t: float) -> float:
    """Comoving radius of the event horizon at time t, ``χ_EH = e^{−Ht}/H``
    (host-side): during inflation it shrinks exponentially — comoving structures do
    not move, the horizon abandons them."""
    return math.exp(-h * t) / h


def mode_crossing_time(k: float, h: float) -> float:
    """When comoving mode k exits the horizon (host-side): ``k = a(t)·H`` with
    ``a = e^{Ht}`` gives ``t_k = ln(k/H)/H``. Larger scales (smaller k) exit EARLIER;
    crossing times are logarithmic in k."""
    return math.log(k / h) / h


def spectral_tilt(epsilon: float, eta: float) -> float:
    """Slow-roll spectral index ``n_s = 1 + 2η − 6ε`` (host-side). ε = η = 0 is exact
    de Sitter (flat, n_s = 1); real inflation rolls, so n_s < 1 — a red tilt."""
    return 1.0 + 2.0 * eta - 6.0 * epsilon


def mode_amplitude(k: float, h: float, n_s: float, k_pivot: float = 1.0) -> float:
    """Frozen super-horizon amplitude for mode k (host-side): the nearly-flat power law
    ``√P(k) ∝ (k/k_pivot)^{(n_s−1)/2}`` normalized to H/2π at the pivot — larger scales
    slightly louder for a red tilt."""
    return (h / (2.0 * math.pi)) * (k / k_pivot) ** (0.5 * (n_s - 1.0))


def efolds(a_start: float, a_end: float) -> float:
    """Number of e-folds ``N = ln(a_end/a_start)`` (host-side); N ≳ 60 solves the
    horizon problem."""
    return math.log(a_end / a_start)


def hubble_flow_redshift(r: float, h: float) -> float:
    """Redshift factor for an emitter comoving at proper distance r inside the horizon
    (host-side, exact static-patch form): ``1 + z = 1/√(1 − H²r²)`` — diverges at the
    horizon, where the emitter freezes and fades exactly like something falling onto a
    black hole, watched from outside."""
    x = max(1.0 - (h * r) * (h * r), 1e-12)
    return 1.0 / math.sqrt(x)
