# The expanding universe, exactly — the scale factor, the Hubble diagram, the two horizons

Friedmann 1922 wrote the equation, Lemaître and Hubble saw the recession, and in
1998 two teams of supernova hunters found distant Type Ia's ~0.25 mag too dim for
any decelerating universe — the expansion is *speeding up*, ~69% of everything is a
cosmological constant, and the discovery took the 2011 Nobel. The punchline nobody
advertises: for the flat matter+Λ universe we actually live in, the Friedmann
equation is *exactly solvable* — this round is built on that closed form, with the
real Planck 2018 numbers asserted throughout. New engine module:
`warp_shaders/engine/cosmology.py`. Scenes: `cosmo_expansion`, `cosmo_hubble`,
`cosmo_horizons`.

## The dictionary

```
(ȧ/a)² = H₀²(Ω_m/a³ + Ω_Λ)               Friedmann, flat, late universe
a(t) = (Ω_m/Ω_Λ)^{1/3} sinh^{2/3}(3/2·√Ω_Λ H₀ t)   the EXACT solution (ODE asserted)
t₀ = (2/3√Ω_Λ H₀)·asinh(√(Ω_Λ/Ω_m)) = 13.8 Gyr     the age (asserted, Planck numbers)
ä = 0 at a = (Ω_m/2Ω_Λ)^{1/3}             acceleration onset, z ≈ 0.63 (asserted)
ρ_m = ρ_Λ at a = (Ω_m/Ω_Λ)^{1/3}          the handoff, z ≈ 0.30 (asserted)
μ(z) = 5·log₁₀(D_L/10pc), D_L = (1+z)D_C   the Hubble diagram (ΛCDM > matter-only, asserted)
D_PH = c∫₀ᵗ dt'/a ≈ 46 Gly                the observable universe (asserted)
D_EH = c∫ₜ^∞ dt'/a ≈ 16.7 Gly             the finite future — pure Λ (asserted)
```

Planck 2018: H₀ = 67.36 km/s/Mpc, Ω_m = 0.3153, flat. Radiation is neglected
(negligible after ~50 Myr; it shifts the particle horizon by ~1 Gly, noted).

## Part I — the scale factor: `cosmo_expansion`

The whole history of expansion is one smooth curve: matter-era `t^{2/3}` early
(gravity braking), de Sitter exponential late (Λ driving), and the closed form
`sinh^{2/3}` seams them exactly — the suite asserts it satisfies the Friedmann ODE
numerically to 10⁻⁶ and that inverting it at a = 1 gives 13.8 Gyr from two measured
numbers and one formula.

The scene plays 32 Gyr: galaxies pinned to comoving positions ride the stretching
grid (nothing moves *through* space), while the panel draws the exact curve past
the gray matter-forever ghost — the universe 1998 expected. The green inflection
dot marks `ä = 0` at z ≈ 0.63 (asserted, including the numeric sign flip of ä):
everything before it decelerates, everything after accelerates. The cyan tick at
13.8 Gyr is us — just past the matter-Λ handoff (z ≈ 0.30, asserted), watching Λ
take the wheel.

## Part II — the Hubble diagram: `cosmo_hubble`

Type Ia supernovae detonate at the Chandrasekhar mass, so they are standard
candles: apparent brightness IS distance. The scene rebuilds the 1998 plot from
the exact machinery — distance modulus `μ = 5·log₁₀(D_L/10pc)` with
`D_L = (1+z)·(c/H₀)∫dz'/E(z')` — as a survey deepening in real time: low-z points
first (Calán/Tololo), then the high-z teams, each supernova flaring as it lands on
the exact ΛCDM curve with the real ±0.15 mag post-correction scatter.

The gray ghost below is the exact matter-only universe (`Ω_m = 1`,
`D_C = (2c/H₀)(1−1/√(1+z))` in closed form) — the fit everyone expected. The suite
asserts ΛCDM sits above it at every z, with the gap reaching 0.58 mag at z = 1:
distant supernovae are ~0.25-0.6 mag dimmer than a coasting-or-braking universe
allows, because the expansion accelerated under them while their light was in
flight. The green divide at z ≈ 0.63 splits the diagram into the decelerating era
(beyond) and the accelerating one (inside) — the shape of the curve across that
divide is the discovery.

## Part III — the two horizons: `cosmo_horizons`

Λ gives the universe a causal structure with two distinct edges, both exact
integrals of the closed-form a(t):

* the **particle horizon** `c∫₀ᵗ dt'/a` — how far light has *ever* been able to
  come: ≈ 46 Gly comoving today (asserted; 3.3× the naive c·t₀, because space
  stretched behind the photons) — the radius of the observable universe;
* the **event horizon** `c∫ₜ^∞ dt'/a` — how far light leaving *now* can ever
  reach: FINITE, ≈ 16.7 Gly (asserted), purely because of Λ. In a matter-only
  universe the integral diverges and everything is eventually reachable.

The scene plays 2 → 60 Gyr in comoving coordinates: the cyan particle horizon
grows (galaxies fade into view as their first light arrives), while the magenta
event horizon shrinks — each galaxy it sweeps past flares amber and goes red:
from that moment nothing we send will ever reach them, and their newest light will
never reach us; we keep receiving their past, redshifted into a frozen goodbye.
The suite asserts both real numbers, that the particle horizon only grows, and
that the comoving event horizon only shrinks. In the far future only the local
group remains — the sky is slowly emptying.

## Sources

- A. Friedmann, *Über die Krümmung des Raumes*, Z. Phys. 10, 377 (1922)
- E. Hubble, *A relation between distance and radial velocity among extra-galactic
  nebulae*, PNAS 15, 168 (1929)
- A. G. Riess et al., *Observational evidence from supernovae for an accelerating
  universe and a cosmological constant*, AJ 116, 1009 (1998)
- S. Perlmutter et al., *Measurements of Ω and Λ from 42 high-redshift
  supernovae*, ApJ 517, 565 (1999)
- Planck Collaboration, *Planck 2018 results. VI. Cosmological parameters*, A&A
  641, A6 (2020) — H₀ = 67.36, Ω_m = 0.3153
- T. M. Davis, C. H. Lineweaver, *Expanding confusion: common misconceptions of
  cosmological horizons*, PASA 21, 97 (2004) — the definitive horizons paper
- D. W. Hogg, *Distance measures in cosmology*, arXiv:astro-ph/9905116 — the
  distance-modulus machinery
