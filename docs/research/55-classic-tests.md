# The classic tests — Mercury's perihelion, the late echo, and the clocks

The tests that made general relativity: Le Verrier logged Mercury's unexplained
43″/century in 1859; Einstein recovered it in November 1915 and was "beside himself
with joyous excitement"; Eddington's 1919 eclipse made him famous; Shapiro added the
fourth test in 1964 — radar echoes past the Sun return late; and Pound-Rebka weighed
a photon in a Harvard stairwell in 1959. Today the whole set runs silently inside
every GPS receiver. It all lives in ONE metric — Schwarzschild — and it is all
closed-form. New engine module: `warp_shaders/engine/geodesics.py` (all laws
test-asserted). Scenes: `gr_precession`, `gr_shapiro`, `gr_clocks`.

## The dictionary

```
V²(r) = (1−2M/r)(1+L²/r²)             the effective potential — the solar system on one curve
L² = Mr²/(r−3M), E = (1−2M/r)/√(1−3M/r)   circular orbits: V′ = 0 exactly (asserted)
r_ISCO = 6M, E = √(8/9), L = 2√3·M    stability's inner edge (asserted)
r_ph = 3M, b_c = 3√3·M                light's last orbit; the EHT shadow
u″ + u = M/L² + 3Mu²                  the Binet equation — 3Mu² IS general relativity
Δφ = 6πM/(a(1−e²))                    Einstein 1915 — Mercury: 42.98″/century (asserted)
α = 4M/b                              Eddington 1919 — 1.75″ at the limb (asserted)
Δt = 4M·[ln(4r₁r₂/b²) + 1]            Shapiro 1964 — Earth-Mars ≈ 250 μs (asserted)
dτ/dt = √(1−2M/r) / √(1−3M/r)         clocks: static / circular orbit
r× = 3R/2                             break-even orbit — mass-INDEPENDENT (asserted)
GPS: +38.5 μs/day                     asserted; Pound-Rebka: gh/c² = 2.46×10⁻¹⁵ (asserted)
```

## Part I — the rosette: `gr_precession`

The scene integrates the EXACT bound geodesic live — RK4 on the Binet equation —
in a strong field (a = 26M, e = 0.5) where each orbit visibly advances its
perihelion: the ellipse becomes a rosette. Newton's closed ellipse ghosts behind it
in gray (drop the `3Mu²` term and the integration returns to it); amber dots mark
each perihelion passage walking around the mass; the body runs on physical pacing
(dτ/dφ = r²/L — it sprints through perihelion and hangs at aphelion); and the
photon sphere and ISCO rim the center — strong-field furniture Mercury never sees
but this orbit skims.

The ledger tells the quantitative story: the amber bar accumulates the measured
advance, and the cyan tick marks Einstein's first-order `6πM/(a(1−e²))` — in this
strong field the exact orbit OVERSHOOTS the formula (asserted: the suite integrates
both regimes; in the weak field, formula and integration agree to better than 2%,
and `mercury_precession_arcsec_century()` returns 42.98″, asserted to 0.1″). The
suite also asserts circular orbits sit exactly at `V′ = 0`, and the ISCO anchors
`E = √(8/9)` — the 5.7% of rest mass an accretion disk shines away.

## Part II — the late echo: `gr_shapiro`

Shapiro's "fourth test": light near a mass travels with coordinate speed
`c(r) = 1 − 2M/r < 1`, so a radar pulse to a planet at superior conjunction returns
late — for Earth-Mars grazing the solar limb, by ≈ 250 μs on a ~40-minute round
trip (asserted; Viking measured it to 0.1%, Cassini's 2002 radio link to 10⁻⁵ —
still the tightest solar-system test of GR).

The scene plays one conjunction: the planet slides behind the Sun, the radar path
bows toward the mass on the exact `4M/b` scale and grazes the corona, and the pulse
VISIBLY slows as it wades through the well — its animation speed is the metric's
`1 − 2M/r`. The panel below draws the exact delay curve of the whole sweep — the
logarithmic spike as b crosses the limb — with a live marker riding it, and the
link blacks out through the occultation window (real superior-conjunction radio
blackouts, dramatized). The suite asserts the Earth-Mars number, monotonicity in b,
and the coordinate slowing itself.

## Part III — the clock lattice: `gr_clocks`

The oldest prediction (Einstein 1907, eight years before the field equations) as a
working instrument: a clock standing at r ticks at `√(1−2M/r)`; a clock on a
circular orbit ticks at `√(1−3M/r)` — altitude blueshift and orbital time dilation
in one exact term. Four real clocks run around one planet:

* the Pound-Rebka tower pair — the upper clock measurably fast (gh/c² = 2.46×10⁻¹⁵
  over Harvard's 22.5 m, asserted);
* a low orbiter BELOW the dashed break-even ring — speed wins, its clock runs SLOW
  (ISS astronauts genuinely age less);
* a high orbiter ABOVE the ring — altitude wins, its clock runs FAST: +38.5 μs/day
  for the real GPS constellation (asserted; uncorrected, fixes would drift ~11
  km/day — general relativity as an engineering requirement);
* the break-even ring itself at `r = 3R/2`, where the two rates cancel EXACTLY and
  — the suite asserts this across three different masses — independently of M:
  pure geometry deciding whether a clock gains or loses.

Hands tick at the exact rates (divergence amplified for the eye; the numbers are
real), and the ledger accumulates the GPS gain and the ISS loss live.

## Sources

- U. Le Verrier, *Théorie du mouvement de Mercure*, Ann. Obs. Paris 5 (1859) — the
  43″ anomaly
- A. Einstein, *Erklärung der Perihelbewegung des Merkur aus der allgemeinen
  Relativitätstheorie*, Sitzungsber. Preuss. Akad. Wiss. 831 (1915)
- A. S. Eddington et al., *A determination of the deflection of light by the sun's
  gravitational field*, Phil. Trans. R. Soc. A 220, 291 (1920)
- I. I. Shapiro, *Fourth test of general relativity*, PRL 13, 789 (1964); Viking:
  Reasenberg et al., ApJ 234, L219 (1979); Cassini: Bertotti, Iess, Tortora,
  Nature 425, 374 (2003)
- R. V. Pound, G. A. Rebka, *Gravitational red-shift in nuclear resonance*, PRL 3,
  439 (1959)
- N. Ashby, *Relativity in the Global Positioning System*, Living Rev. Relativ. 6,
  1 (2003) — the +38 μs/day budget
- S. Chandrasekhar, *The Mathematical Theory of Black Holes* (Oxford 1983) — the
  geodesic structure, ISCO, photon sphere
