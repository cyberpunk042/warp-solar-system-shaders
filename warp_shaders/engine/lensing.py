"""The gravitational-lensing dictionary (host-side): light bent, exactly.

Eddington 1919 measured the deflection; Einstein 1936 wrote down the point-lens ring;
Walsh-Carswell-Weymann 1979 found the first doubled quasar; Paczyński 1986 turned the
light curve into a dark-matter probe; Refsdal 1964 showed the time delay measures the
Hubble constant. The point-mass lens is EXACTLY solvable — every law below is closed
form and test-asserted (angles in units of the Einstein radius where convenient):

* ``einstein_radius`` — perfect alignment smears the source into a ring of angular
  radius ``θ_E = √(4M·D_LS/(D_L·D_S))`` — the natural unit of every lens.
* ``image_positions`` — the lens equation ``β = θ − θ_E²/θ`` is a quadratic: ALWAYS
  two images,

      θ± = (β ± √(β² + 4θ_E²))/2,

  one outside the ring (the minimum), one inside, inverted (the saddle). Both
  asserted to satisfy the lens equation exactly.
* ``magnifications`` — surface brightness is conserved, so images (de)magnify by the
  Jacobian: ``μ = 1/(1 − (θ_E/θ)⁴)``. The signed magnifications obey the beautiful
  point-lens sum rule

      μ₊ + μ₋ = 1        (asserted exactly)

  — however the source moves, the SIGNED flux ledger balances to one source.
* ``paczynski_magnification`` — unresolved images add in absolute value:

      A(u) = (u² + 2)/(u·√(u² + 4)),   u = β/θ_E

  — the microlensing light curve: A → 1 far away, A(1) = 3/√5, A → 1/u diverging at
  perfect alignment (all asserted). Symmetric, achromatic, unrepeatable: the
  signature OGLE/MOA hunt for — and how exoplanets and dark compact objects are found.
* ``fermat_potential`` — the arrival-time surface

      τ(θ) = (θ − β)²/2 − θ_E²·ln|θ|

  (geometric delay + Shapiro delay). **Images sit at stationary points of arrival
  time** — Fermat's principle in curved spacetime: dτ/dθ = 0 exactly at θ± (asserted
  numerically). θ₊ is the minimum, θ₋ the saddle.
* ``time_delay`` — τ(θ₋) − τ(θ₊) > 0 (asserted): the saddle image arrives LATE.
  Scaled by distances this is Refsdal's cosmography — measured lags between quasar
  images (e.g. ~417 days in the first double QSO 0957+561) weigh the universe.

See ``docs/research/54-gravitational-lensing.md``.
"""

import math


def einstein_radius(mass: float, d_l: float, d_s: float) -> float:
    """The Einstein radius (host-side): ``θ_E = √(4M·D_LS/(D_L·D_S))`` with
    ``D_LS = D_S − D_L`` (flat geometry) — the angular scale every lens is measured in."""
    return math.sqrt(4.0 * mass * (d_s - d_l) / (d_l * d_s))


def image_positions(beta: float, theta_e: float):
    """Solve the point-lens equation ``β = θ − θ_E²/θ`` exactly (host-side):
    ``θ± = (β ± √(β²+4θ_E²))/2`` — always two images, θ₊ outside the ring, θ₋ inside
    and on the far side (negative). Both satisfy the lens equation (asserted)."""
    disc = math.sqrt(beta * beta + 4.0 * theta_e * theta_e)
    return (0.5 * (beta + disc), 0.5 * (beta - disc))


def lens_equation(theta: float, theta_e: float) -> float:
    """Where the ray at image angle θ came from (host-side): ``β = θ − θ_E²/θ`` —
    the exact deflection map the inverse ray-shooting scene runs per pixel."""
    return theta - theta_e * theta_e / theta


def magnification(theta: float, theta_e: float) -> float:
    """Signed image magnification (host-side): ``μ = 1/(1 − (θ_E/θ)⁴)`` — the inverse
    Jacobian of the lens map. Negative for the inner (parity-flipped, saddle) image."""
    r = theta_e / theta
    return 1.0 / (1.0 - r ** 4)


def magnifications(beta: float, theta_e: float):
    """Both signed magnifications (host-side) — the point-lens sum rule
    ``μ₊ + μ₋ = 1`` holds for every β (asserted exactly)."""
    tp, tm = image_positions(beta, theta_e)
    return (magnification(tp, theta_e), magnification(tm, theta_e))


def paczynski_magnification(u: float) -> float:
    """The microlensing light curve (host-side): ``A(u) = (u²+2)/(u√(u²+4))`` — the
    unresolved sum |μ₊|+|μ₋|. A(∞)=1, A(1)=3/√5, A→1/u at alignment (asserted)."""
    return (u * u + 2.0) / (u * math.sqrt(u * u + 4.0))


def fermat_potential(theta: float, beta: float, theta_e: float) -> float:
    """The arrival-time surface (host-side): ``τ(θ) = (θ−β)²/2 − θ_E²ln|θ|`` —
    geometric plus Shapiro delay. Images live at its stationary points (asserted)."""
    return 0.5 * (theta - beta) ** 2 - theta_e * theta_e * math.log(abs(theta))


def fermat_potential_2d(tx: float, ty: float, bx: float, by: float,
                        theta_e: float) -> float:
    """The 2D arrival-time landscape (host-side/kernel-mirrored): ``τ(θ) =
    |θ−β|²/2 − θ_E²ln|θ|`` — the surface the ``lens_fermat`` scene draws, whose
    minimum and saddle ARE the two images."""
    r = math.sqrt(tx * tx + ty * ty)
    return 0.5 * ((tx - bx) ** 2 + (ty - by) ** 2) - theta_e * theta_e * math.log(max(r, 1e-12))


def time_delay(beta: float, theta_e: float) -> float:
    """Relative arrival delay between the two images (host-side):
    ``Δτ = τ(θ₋) − τ(θ₊) > 0`` — the saddle image arrives late (asserted). Scaled by
    ``(1+z_L)·D_LD_S/D_LS`` this is Refsdal's H₀ measurement."""
    tp, tm = image_positions(beta, theta_e)
    return fermat_potential(tm, beta, theta_e) - fermat_potential(tp, beta, theta_e)
