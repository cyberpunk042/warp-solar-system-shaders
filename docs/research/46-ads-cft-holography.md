# Research 46 — AdS/CFT: holographic duality made visible

> The relativity set (research 42–43) rendered *solutions* of general relativity — holes,
> binaries, wormholes. This note renders an idea **about** gravity itself: the **holographic
> principle**, in its sharpest known form, the **AdS/CFT correspondence**. Nothing here is a
> mood-board — every element of the `ads_cft` frame is one term of the duality dictionary, and
> the tiling geometry is derived (not tuned) from hyperbolic trigonometry.

## The physics — a universe on the boundary

**Maldacena's conjecture** (1997, [hep-th/9711200](https://arxiv.org/abs/hep-th/9711200); made
precise by Gubser–Klebanov–Polyakov [hep-th/9802109](https://arxiv.org/abs/hep-th/9802109) and
Witten [hep-th/9802150](https://arxiv.org/abs/hep-th/9802150)): a theory of quantum gravity in a
(d+1)-dimensional **Anti-de Sitter** spacetime — the maximally symmetric solution of Einstein's
equations with *negative* cosmological constant — is exactly equivalent to a **conformal field
theory** with no gravity at all, living on the d-dimensional boundary of that spacetime. The
bulk is the hologram's image; the boundary theory is the film. It is the most cited result in
high-energy physics and the concrete realization of 't Hooft's and Susskind's holographic
principle ([gr-qc/9310026](https://arxiv.org/abs/gr-qc/9310026),
[hep-th/9409089](https://arxiv.org/abs/hep-th/9409089)).

A constant-time slice of AdS₃ is the **hyperbolic plane** H², and its conformal compactification
is the **Poincaré disk**: the infinite negatively curved plane mapped into a finite Euclidean
circle, angles preserved, distances diverging toward the rim as `ds = 2|dz|/(1 − |z|²)`. The rim
`|z| = 1` is *infinitely far away* in hyperbolic distance yet finitely drawn — that rim **is**
the conformal boundary where the CFT lives. This is why the Poincaré disk is the canonical
cartoon of AdS/CFT, and why the scene is built on it.

## What each element of the frame is

| Visual element | Duality dictionary term |
|---|---|
| `{7,3}` heptagon tiling filling the disk | The bulk: identical hyperbolic cells, crowding at the rim = pure metric divergence (the "UV" of the boundary theory) |
| Glowing ring at `r = 1` | The conformal boundary — spatial infinity of AdS, home of the CFT |
| Same tiling outside, warm-tinted, through `z → z/\|z\|²` | The hologram: the inversion maps the exterior conformally onto the interior — boundary data encoding the bulk, drawn literally |
| Bright circular arcs orthogonal to the rim | **Ryu–Takayanagi geodesics** ([hep-th/0603001](https://arxiv.org/abs/hep-th/0603001)): the entanglement entropy of a boundary interval equals the length of the bulk geodesic anchored on its endpoints, `S = L/4G` |
| Dots where arcs meet the rim | The interval endpoints — boundary operator insertions |
| The whole tiling flowing (`--frames`) | A **Möbius isometry** of the bulk `z → (z − a)/(1 − āz)`, which acts on the rim as a conformal transformation — the entry-one of the dictionary: bulk isometries ↔ boundary conformal symmetries |

## The tiling — a reflection group, derived not tuned

The `{p,q}` tiling (p-gons, q around each vertex) is hyperbolic whenever
`1/p + 1/q < 1/2` — `{7,3}` is the classic minimal case, the geometry of Escher's *Circle
Limit* prints (which Coxeter famously annotated). The renderer folds every pixel into one
fundamental wedge of the symmetry group:

1. rotate into the wedge `|arg z| ≤ π/7` (fold the 7-fold rotation),
2. reflect across the x-axis (`y → |y|`),
3. if inside the edge-mirror circle, **invert** in it (`w → w·r²/|w|²` about its centre) —
   inversion in a circle orthogonal to the unit circle is a hyperbolic reflection,
4. repeat until the point escapes the mirror (≤ 48 folds).

The mirror circle is fixed by hyperbolic trigonometry, no magic numbers. In the right triangle
(polygon centre O, edge midpoint M, vertex V) with angles `π/p` at O and `π/q` at V, the
hyperbolic leg OM obeys

```
cosh m = cos(π/q) / sin(π/p)
```

so the Euclidean distance of the edge from the origin is `x₀ = tanh(m/2)`, and the geodesic
through it perpendicular to the axis is the circle with centre `d = (1 + x₀²)/(2x₀)` and radius
`r = d − x₀` (which satisfies the orthogonality condition `d² = 1 + r²` identically). The
reflection count doubles as an **orbit trap**: its parity chequers the cells, its magnitude is
the "generation" (hyperbolic distance from the origin), and past ~14 folds the cells are
sub-pixel — the frame fades them into rim shimmer instead of aliasing, which is exactly the
right physics metaphor (the boundary is the UV limit of the bulk).

Anti-aliasing under a conformal map needs the **pixel footprint** carried through every
transformation: the exterior inversion scales it by `1/r²`, the Möbius flow by its Jacobian
`|f′(z)| = (1 − |a|²)/|1 − āz|²`, and every mirror inversion by `r²/|w|²`. Edge lines are then
drawn at constant *screen* width from the folded-space distance divided by the accumulated
scale — the same trick the fractal scenes use for distance-estimator colouring.

## The RT geodesics

For boundary endpoints `u, v` on the unit circle, the unique circle through both and orthogonal
to the rim has centre `c = (u + v)/(1 + u·v)` and radius `√(|c|² − 1)` — orthogonality falls out
identically since `c·u = c·v = 1`. Its arc inside the disk is the hyperbolic geodesic: the
Ryu–Takayanagi "minimal surface" of the interval (in 2+1 bulk dimensions a minimal surface is a
geodesic). Three intervals drift and breathe along the boundary; their faint mirrored arcs
continue into the exterior hologram at reduced weight.

## Part II — the engine-level bulk: `ads_bulk` and `engine/adscft.py`

The disk scene is the *map*; `ads_bulk` is the *territory*. The camera floats **inside**
global AdS with a Schwarzschild-AdS black hole at the centre, and every camera ray is a photon
integrated through the curved bulk — the same honesty standard as gargantua/kerr (research
42–43), extended with what only AdS has: a boundary you can reach.

### Null geodesics: Λ drops out of the path shape

The Schwarzschild-AdS metric has blackening factor `f(r) = 1 + r²/L² − 2M/r`. For a null
geodesic with impact parameters folded into `u(φ) = 1/r(φ)`, the orbital equation is

```
d²u/dφ² + u = 3Mu²
```

— *identical* to pure Schwarzschild: the cosmological-constant term contributes a constant to
the effective potential and **drops out of the photon path shape** (Islam 1983, *"The
cosmological constant and classical tests of general relativity"*). So `ads_bulk` reuses the
proven null-geodesic pull `a = −(3/2)h²x/r⁵` from `engine/blackhole.py` verbatim — shadow,
photon ring and disk lensing are all real, and all shared with the relativity set.

### What Λ does change: the AdS box

In AdS the conformal boundary is **timelike** and sits at finite optical distance — an
outgoing light ray reaches it (in finite coordinate time) and, with the standard reflecting
boundary conditions of AdS, **comes back in**. The renderer implements exactly that: when a
ray crosses the cutoff sphere `r = R_bdy` it deposits the boundary's CFT emission, mirrors its
velocity about the radial normal (angular momentum `h = |x × v|` is preserved by the mirror,
as the tangential component is untouched), and continues. The number of bounces scales with
`--quality` (low 1 → ultra 4), so higher tiers show the hole and its disk **re-imaged in the
boundary mirror** — AdS as a resonant box, the geometric reason bulk physics is recorded on
the boundary.

### The CFT on the boundary, thermal at the Hawking temperature

Every boundary hit is textured by `engine.adscft.boundary_cft`: the hit direction is
stereographically projected (a conformal map, so the pattern is a genuine conformal field on
the sphere) into the **same `{7,3}` reflection-group fold** the disk scene uses —
`poincare_fold` is one shared `@wp.func`, one fold serving both duals. On top of the lattice
rides a thermal wash proportional to the hole's **Hawking temperature**

```
T = f'(r_h) / 4π ,      f(r_h) = 0
```

computed host-side by bisection (`hawking_temperature`). This is the Hawking–Page dictionary
entry drawn literally: *a black hole in the bulk is a thermal state of the boundary theory*,
and large AdS holes get hotter with size (positive specific heat) — the thermodynamic stability
behind the deconfinement interpretation.

Emission divergences at the true boundary (`f → ∞`) are avoided by defining the CFT glow in
boundary-frame units at the finite cutoff `R_bdy` — which is precisely the regularization
scheme of **holographic renormalization**. Bulk disk emission additionally carries the AdS
redshift ratio `√(f(r_em)/f(r_cam))` between emission point and camera.

## Sources

- J. Maldacena, *The Large N Limit of Superconformal Field Theories and Supergravity*,
  [hep-th/9711200](https://arxiv.org/abs/hep-th/9711200)
- S. Gubser, I. Klebanov, A. Polyakov, *Gauge Theory Correlators from Non-Critical String
  Theory*, [hep-th/9802109](https://arxiv.org/abs/hep-th/9802109)
- E. Witten, *Anti De Sitter Space And Holography*,
  [hep-th/9802150](https://arxiv.org/abs/hep-th/9802150)
- S. Ryu, T. Takayanagi, *Holographic Derivation of Entanglement Entropy from AdS/CFT*,
  [hep-th/0603001](https://arxiv.org/abs/hep-th/0603001)
- G. 't Hooft, *Dimensional Reduction in Quantum Gravity*,
  [gr-qc/9310026](https://arxiv.org/abs/gr-qc/9310026); L. Susskind, *The World as a
  Hologram*, [hep-th/9409089](https://arxiv.org/abs/hep-th/9409089)
- H. S. M. Coxeter, *Crystal Symmetry and Its Generalizations* (1957) — the `{p,q}` hyperbolic
  tilings; M. C. Escher, *Circle Limit I–IV* (1958–60) — the visual precedent
- J. W. Anderson, *Hyperbolic Geometry* (Springer) — Poincaré-disk model, geodesics as circles
  orthogonal to the boundary, hyperbolic right-triangle relations
- S. W. Hawking, D. N. Page, *Thermodynamics of Black Holes in Anti-de Sitter Space*,
  Commun. Math. Phys. 87 (1983) 577 — Hawking temperature of AdS holes, phase transition
- J. N. Islam, *The cosmological constant and classical tests of general relativity*,
  Phys. Lett. A 97 (1983) 239 — Λ drops out of the null-geodesic orbital equation
- O. Aharony, S. Gubser, J. Maldacena, H. Ooguri, Y. Oz, *Large N Field Theories, String
  Theory and Gravity* (the MAGOO review), [hep-th/9905111](https://arxiv.org/abs/hep-th/9905111)
  — reflecting boundary conditions, AdS-as-a-box, the duality dictionary
