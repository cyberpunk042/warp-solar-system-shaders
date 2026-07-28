# The spinning black hole — dragged space, the energy mine, and the bomb

The holography arc (research 44→49) lived in the AdS box. This set steps outside, to the
black hole the universe actually makes: **Kerr**. Rotation buys three phenomena a static
hole cannot have — an ergosphere, an energy mine, and an instability wild enough that
its discoverers named it a bomb — and all three run on closed forms
(`warp_shaders/engine/kerr.py`, test-asserted). Scenes: `kerr_ergosphere`,
`kerr_penrose`, `kerr_superradiance` (complementing the fixed-spin cinematic `kerr`
scene of [research 43](43-relativistic-masterpieces.md)).

## The dictionary

```
r_± = M ± √(M² − a²)          r_+ + r_− = 2M,  r_+·r_− = a²   (asserted)
r_E(θ) = M + √(M² − a²cos²θ)  touches r_+ at the poles, 2M at the equator
Ω_H = a/(2Mr_+)               the horizon's forced rotation rate
T = √(M² − a²)/(4πMr_+)       → 1/8πM at a = 0, → 0 at extremality (third law)
S = A/4 = 2πMr_+
M_irr = √(Mr_+/2)             Christodoulou: M_irr² ∝ area — it can NEVER decrease
```

## Part I — the ergosphere is born: `kerr_ergosphere`

Between the horizon and the stationary limit `r_E(θ)` lies the **ergoregion**: the
metric's `g_tt` changes sign there, so *no* observer can stand still — spacetime itself
rotates, and everything in it is swept along (Lense–Thirring frame dragging, far-field
rate `ω_LT = 2Ma/r³`, the 1/r³ law asserted). The scene makes the spin the time axis:
χ = a/M sweeps 0 → 0.98 → 0 once per cycle, and

- the horizon `r_+ = M(1 + √(1−χ²))` **contracts** from 2M toward M,
- the violet ergosurface **peels away from it** (they coincide at χ = 0) into the
  oblate shell — rendered as accumulated ray path-length through the ergoregion, so
  escaping grazers paint a violet crescent hugging the shadow,
- the drag (a genuine 1/r³ gravitomagnetic force on the photons — deliberately *not*
  the per-ray-amplified artistic force of the cinematic `kerr` scene, which would
  balloon the veil across the frame) smears the starfield and warps the shadow into
  its lopsided Kerr shape.

## Part II — the Penrose mine: `kerr_penrose`

Penrose 1969 (with Floyd, 1971): the ergoregion admits orbits of **negative total
energy**. Drop a particle in, split it there; one fragment falls in with E < 0, the
other escapes carrying *more* than went in. The hole pays from its spin. Christodoulou
1970 found the ledger: `M² = M_irr² + J²/4M_irr²` — mass is irreducible core plus
rotational store — and **M_irr never decreases** (equivalent, via `A = 16πM_irr²`, to
Hawking's area theorem). The maximum haul from an extremal hole is

```
M − M_irr = M(1 − 1/√2) ≈ 29.3%   —   the Penrose bound
```

The engine mines honestly: `penrose_extract` books each event as
`δJ = −δE/(q·Ω_H)` with reversibility q ≤ 1, substepped so the first law
`dA ∝ dM − Ω_H dJ ≥ 0` holds along the path. The suite asserts **M_irr monotone over
300 random extraction events** and that a near-reversible mine recovers **99.9% of the
bound**. On screen: particles dive, split, the crimson fragments feed the hole, the
green ones leave richer — and after each event the swirl slows and the horizon
**grows** (the suite counts its pixels: the area theorem, watched) until the violet
annulus pinches shut. A bigger, slower, dead hole.

## Part III — the black-hole bomb: `kerr_superradiance`

Zel'dovich 1971: the wave version of Penrose. A mode `e^{−iωt+imφ}` scattering off a
spinning hole is **amplified** iff

```
0 < ω < m·Ω_H
```

(`superradiant`, the exact condition, asserted on both sides of the boundary). The
reflected wave carries away rotational energy — a hole can amplify light. Press &
Teukolsky 1972 added the mirror: trapped between mirror and horizon, the wave
re-amplifies every crossing, `A_n = A₀(1+g)ⁿ` (`bomb_amplitude`) — exponential runaway.
They called the paper *Floating Orbits, Superradiant Scattering and the Black-hole
Bomb*. The same instability, with the mirror replaced by a massive boson's potential
wall, is real astrophysics: spinning holes can spin *down* by growing boson clouds,
which is how black-hole spin measurements constrain ultralight dark-matter candidates
(Brito–Cardoso–Pani).

The scene arms the bomb with an honestly superradiant mode (ω = ½·m·Ω_H), ratchets the
trapped m = 2 spiral up pass by pass while the horizon's spin spokes slow, and lets the
mirror burst at the critical amplitude — flash, shrapnel, and a quieter hole.

## Sources

- R. Kerr, *Gravitational field of a spinning mass*, Phys. Rev. Lett. 11, 237 (1963)
- R. Penrose, R. Floyd, *Extraction of Rotational Energy from a Black Hole*, Nature
  Phys. Sci. 229, 177 (1971)
- D. Christodoulou, *Reversible and Irreversible Transformations in Black-Hole
  Physics*, PRL 25, 1596 (1970) — the irreducible mass
- J. Bardeen, B. Carter, S. Hawking, *The four laws of black hole mechanics*, CMP 31,
  161 (1973)
- Ya. Zel'dovich, *Generation of waves by a rotating body*, JETP Lett. 14, 180 (1971)
- W. Press, S. Teukolsky, *Floating Orbits, Superradiant Scattering and the Black-hole
  Bomb*, Nature 238, 211 (1972)
- R. Brito, V. Cardoso, P. Pani, *Superradiance*,
  [1501.06570](https://arxiv.org/abs/1501.06570) — the modern review
