# Light bent, exactly — arcs, microlensing, and the Fermat landscape

The other great observational triumph of general relativity, and the one that started
it all: Eddington's 1919 eclipse measured the deflection, Einstein 1936 wrote down the
ring "with no hope of observing this phenomenon directly", Walsh–Carswell–Weymann 1979
found the first doubled quasar, and today lensing weighs dark matter, finds exoplanets,
and referees the Hubble tension. The point-mass lens is *exactly solvable* — this
round is built on that closed-form core. New engine module:
`warp_shaders/engine/lensing.py` (all laws test-asserted). Scenes: `lens_arcs`,
`lens_microlensing`, `lens_fermat`.

## The dictionary

```
θ_E = √(4M·D_LS/D_L D_S)             the Einstein radius — every lens's natural unit
β = θ − θ_E²/θ                        the lens equation (exact, per-pixel in lens_arcs)
θ± = (β ± √(β²+4θ_E²))/2              ALWAYS two images (lens eq satisfied, asserted)
μ = 1/(1−(θ_E/θ)⁴)                    signed magnification (inverse Jacobian)
μ₊ + μ₋ = 1                           the point-lens sum rule (asserted exactly)
A(u) = (u²+2)/(u√(u²+4))              Paczyński: |μ₊|+|μ₋| (asserted; A(1)=3/√5)
τ(θ) = |θ−β|²/2 − θ_E²ln|θ|           the Fermat arrival-time surface
dτ/dθ = 0 at θ±                        images ARE stationary points (asserted)
Δτ = τ(θ₋) − τ(θ₊) > 0                the saddle image arrives late (asserted)
```

## Part I — arcs and the ring: `lens_arcs`

Inverse ray shooting is the most shader-native physics in the engine: for every pixel
of the image plane, run the lens equation backwards — `β = θ(1 − θ_E²/|θ|²)` — and
sample the source plane there. Surface brightness is conserved along rays (Liouville),
so that one line *is* gravitational lensing. A spiral galaxy transits behind the lens:
gentle shear becomes a tangential arc outside the Einstein radius, a smaller
parity-flipped counter-image appears inside (the quadratic's second root — always
there), and at alignment the pair wraps into the full **Einstein ring**.

The suite asserts the exact two-image solution against the same lens equation the
kernel runs. And the magnified images are *bigger, not brighter per unit area* —
lensing gathers surface brightness without changing it, which is why clusters work as
nature's telescopes for JWST's deepest sources.

## Part II — the light curve: `lens_microlensing`

Stellar lenses split images by micro-arcseconds — unresolvable — so all you see is the
summed brightness, and it is exactly Paczyński's curve `A(u) = (u²+2)/(u√(u²+4))`.
The suite asserts it equals `|μ₊|+|μ₋|` from the exact image solution, alongside the
point lens's prettiest identity: the **signed** sum `μ₊ + μ₋ = 1`, for every source
position — the flux ledger always balances to one source.

The scene plays a transit: the source slides behind the lens while its two images
(exact `θ±`, brightness ∝ exact `|μ|`) slide along the axis — the outer swelling, the
inner rising toward the ring — as the light curve below reveals live: flat, the smooth
achromatic rise, `A(u₀=0.25) ≈ 4.1` at closest approach, the symmetric fall.
Achromatic, symmetric, unrepeatable: the signature that separates lensing from every
variable star. Paczyński proposed the experiment in 1986; OGLE and MOA have logged
tens of thousands of events and found planets by the kink a companion adds to exactly
this curve.

## Part III — the arrival-time landscape: `lens_fermat`

The deepest picture (Schneider 1985, Blandford–Narayan 1986): light takes all paths,
and images form where the arrival time is stationary — Fermat's principle in curved
spacetime. The surface `τ(θ) = |θ−β|²/2 − θ_E²ln|θ|` is the geometric delay (bent
paths are longer) plus the Shapiro delay (clocks run slow in the well). The scene
draws it as live contours — a paraboloid dented by the logarithmic funnel — with the
cyan image in the valley (the minimum: first light) and the magenta image on the pass
(the saddle: late), at the exact `θ±`. The suite asserts `dτ/dθ = 0` at both images,
the saddle's negative tangential curvature, and `Δτ > 0`.

The delay is measurable — QSO 0957+561's two images flicker in the same pattern 417
days apart — and Refsdal 1964 saw what it buys: the lag scales with the *absolute
size* of the universe. Time-delay cosmography (H0LiCOW/TDCOSMO) now turns lensed
quasars and supernovae into few-percent measurements of H₀, refereeing the Hubble
tension with nothing but geometry and patience.

## Sources

- A. S. Eddington et al., *A determination of the deflection of light by the sun's
  gravitational field*, Phil. Trans. R. Soc. A 220, 291 (1920) — the 1919 eclipse
- A. Einstein, *Lens-like action of a star by the deviation of light*, Science 84,
  506 (1936)
- D. Walsh, R. F. Carswell, R. J. Weymann, *0957+561 A,B: twin quasistellar objects
  or gravitational lens?*, Nature 279, 381 (1979)
- S. Refsdal, *On the possibility of determining Hubble's parameter from the
  gravitational lens effect*, MNRAS 128, 307 (1964)
- B. Paczyński, *Gravitational microlensing by the galactic halo*, ApJ 304, 1 (1986)
- R. Blandford, R. Narayan, *Fermat's principle, caustics, and the classification of
  gravitational lens images*, ApJ 310, 568 (1986)
- P. Schneider, J. Ehlers, E. Falco, *Gravitational Lenses* (Springer 1992) — the
  standard reference
