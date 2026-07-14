# Research 43 — Relativistic masterpieces (spinning, binary & wormhole geodesics)

> Research 42 built **gargantua**: a Schwarzschild black hole where every camera ray is a photon
> integrated through curved spacetime, no analytic lensing trick. This note takes that same
> honesty and pushes it into three harder geometries — a **spinning** hole, a **binary** pair
> spiralling to merger, and a traversable **wormhole** — all sharing one core so the set stays
> coherent. Nothing here is painted on; the images fall out of integrating photon paths.

## The shared core — `engine/blackhole.py`

Gargantua's disk shader and background were extracted verbatim into two reusable `@wp.func`s so
every hole in the set is lit by the *same* physics:

- **`disk_emission(cp, pdir, time, r_in, r_out, temp0, bright)`** — a thin equatorial accretion
  disk sampled where a ray crosses `y = 0`: a Shakura–Sunyaev `T ∝ r^-3/4` blackbody gradient,
  **relativistic Doppler beaming** (`∝ D³`, the orbital side coming toward you brightens and
  blueshifts), **gravitational redshift** `√(1 − r_s/r)`, and turbulent Keplerian banding. The
  `temp0`/`bright`/`r_in`/`r_out` parameters let each scene set its own look while the maths is
  shared.
- **`cosmic_background(rd, mw)`** — the lensed starfield (+ optional Milky-Way band) read in the
  ray's *final, bent* direction.

gargantua was refactored onto `disk_emission` and verified **pixel-identical** (max abs diff
`0.0`) — the shared core is a true extraction, not a rewrite.

## Kerr — a spinning hole and frame-dragging

A real astrophysical black hole rotates, and rotation drags spacetime around with it — the
**Lense–Thirring effect**. A spinning mass has a *gravitomagnetic* field, the gravitational
analogue of a magnetic dipole:

```
B_g(x)  ∝  [ 3 (Ĵ·r̂) r̂  −  Ĵ ] / r³            (spin Ĵ along the disk axis)
```

and a moving photon feels a force `a = κ · v × B_g`, exactly like the Lorentz force. We add this
to gargantua's Schwarzschild pull:

```
a  =  −(3/2) h² x / r⁵   +   κ · v × B_g
```

Two things become visible:

- **The shadow skews.** Photons co-rotating with the hole (the *prograde* side) are swept inward
  and captured more easily; counter-rotating photons are flung wide. The black shadow loses its
  circular symmetry and flattens on one edge — the asymmetric **"D"** that is the fingerprint of a
  Kerr hole, and (at high spin) the reason the Event Horizon Telescope's images are lopsided.
- **One-sided Doppler.** Spin lets the innermost stable orbit sit closer and move faster, so the
  approaching inner edge is beamed into a brilliant blue-white blade while the receding edge sinks
  to a dim ember — far more lopsided than a static hole.

This is a *visual* Kerr, not the exact Kerr metric (which has no closed-form Cartesian geodesic);
the gravitomagnetic term reproduces frame-dragging's qualitative signature faithfully and cheaply.

## Binary black hole — two shadows, eyeholes, and a chirp

Two Schwarzschild holes orbit their common centre. There is no closed-form two-body metric, but
light deflection superposes well at the separations that matter here: each photon feels the sum of
two single-centre null-geodesic pulls,

```
a  =  Σ_k  −(3/2) h_k² (x − c_k) / r_k⁵ ,     h_k = |(x − c_k) × v|
```

so near either hole the deflection is that hole's *exact* Schwarzschild bending. The payoff is the
imaging that numerical-relativity groups (Bohn et al. 2015, "What does a binary black hole
merger look like?") made famous:

- **Two shadows that lens each other.** Each hole warps the starfield *and* carries a distorted
  little copy of its companion around its rim — the "**eyeholes**" effect.
- **Einstein photon-rings.** Rays that graze a photon sphere (`r ≈ 1.5 r_s`) pile up into a bright
  ring around each shadow; we paint this as a hot blue-white halo accumulated along the path.
- **The chirp.** Over frames the separation shrinks and the orbit whirls faster — the runaway
  **inspiral** that ends in merger. A quadrupole **gravitational-wave** shear
  `d → d·(1 ± ε(d_x² − d_z²))` ripples the lensed background outward as the pair tightens: the
  spacetime distortion LIGO first heard on 14 September 2015, here made visible. At the end the
  two shadows wheel together into one.

## Wormhole dive — a geodesic through an Ellis throat

The **Ellis / Morris–Thorne** wormhole is the textbook traversable tunnel:

```
ds²  =  −dt²  +  dℓ²  +  (b₀² + ℓ²)(dθ² + sin²θ dφ²)
```

The signed coordinate `ℓ` runs from `−∞` (our universe) through `0` (the throat) to `+∞`
(another universe); the areal radius `r = √(b₀² + ℓ²)` never falls below the throat radius `b₀`,
so the tunnel never pinches shut. Because the metric is spherically symmetric, each photon stays
in a plane and carries a conserved angular momentum `L`; its null geodesic reduces to

```
ℓ''  =  L² ℓ / r⁴ ,        φ'  =  L / r² ,        p_ℓ² + L²/r² = 1
```

which we integrate per ray. The impact parameter decides the fate:

- **Small `L` (small impact parameter): threads the throat.** `ℓ` runs through `0` to `+∞` — the
  ray comes out the *other side*, so a whole second universe is fish-eyed into the disc in the
  middle of the frame.
- **Large `L`: turns back.** There is a turning point at `r = L` (`ℓ = ±√(L² − b₀²)`) and the ray
  returns to our universe, lensing our own sky into a bright **Einstein ring** around the
  exotic-matter mouth.

We give the two universes different palettes (our cool blue Milky-Way band vs. an amber one) so
the portal reads instantly, and over `--frames` we slide the camera's `ℓ` from deep in our
universe, through the throat, and out the far side — a genuine **fly-through** in which the second
universe swells from a coin to the whole sky and the mouth flips to show *our* universe behind
you. James et al. 2015 ("Visualizing Interstellar's Wormhole") is the reference for this class of
render.

## Why these are honest

Every image in this set comes from marching photons, not from an SDF of a lens or a texture on a
sphere:

| Scene | Metric / model | Signature that falls out of the physics |
|---|---|---|
| `gargantua` | Schwarzschild null geodesic | disk lensed up-and-over the shadow; photon ring; Doppler + redshift |
| `kerr` | Schwarzschild + Lense–Thirring `v×B_g` | asymmetric "D" shadow; one-sided blue-white beamed edge |
| `binary_bh` | superposed two-centre geodesics | twin lensing shadows ("eyeholes"); Einstein rings; GW shear; inspiral |
| `wormhole_dive` | Ellis / Morris–Thorne geodesic | second universe through the throat; Einstein ring; traversal |

Because the integrators are deterministic (no Monte-Carlo), the renders are fast enough to sweep
into orbit and inspiral GIFs — the motion is where the relativity really lands.

## Sources

- A. Bohn, W. Throwe, F. Hébert, K. Henriksson, D. Bunandar, M. Scheel, N. Taylor,
  *"What does a binary black hole merger look like?"*, Class. Quantum Grav. 32 (2015) 065002 —
  superposed-lensing binary imaging, the "eyeholes".
- O. James, E. von Tunzelmann, P. Franklin, K. Thorne,
  *"Visualizing Interstellar's Wormhole"*, Am. J. Phys. 83 (2015) 486 — Ellis-throat and disk
  ray-tracing that inspired this set.
- M. Morris & K. Thorne, *"Wormholes in spacetime and their use for interstellar travel"*,
  Am. J. Phys. 56 (1988) 395 — the traversable-wormhole metric.
- J. Lense & H. Thirring (1918) — frame-dragging by a rotating mass.
- B. P. Abbott et al. (LIGO/Virgo), *"Observation of Gravitational Waves from a Binary Black Hole
  Merger"*, Phys. Rev. Lett. 116 (2016) 061102 — GW150914, the merger chirp.
- Research 42 (`docs/research/42-gravitational-lensing.md`) — the Schwarzschild geodesic this set
  builds on.
