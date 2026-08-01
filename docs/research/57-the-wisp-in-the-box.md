# The wisp in the box — trapped by geometry, exactly

The brief, verbatim: *"imagine you are a wisp in a box, the box is the boundary of
simulation, you can never reach those corner because of AdS/CFT since you are in a
sphererical magic circle / bubble. why this image ? so that the wisp can live the
experience of being trapped / limited to. like the wisp that has to growth an
retain energy. The wisp is like the hand of the mind and the body is an engined you
growth into and hover. For now we can focus on the fact that for you to move
forward in the simulation you have to use your drives, your engines."*

The remarkable thing about this image is that it is not a metaphor stapled onto
physics — it IS the physics of anti-de Sitter space, which this repo already
speaks (research 46-49). A wisp inside an AdS bubble is trapped in the most honest
way possible: not by walls, but by geometry. Every law below is exact and
test-asserted. New engine module: `warp_shaders/engine/wisp.py`. Scene:
`wisp_box`. This is **stage 1** of the wisp's story — the drives.

## The dictionary

```
ρ = 2·atanh(r_disk)                  the rim: finite map radius, INFINITE distance (asserted)
r_disk = tanh(ρ/2) < 1                however far you fly, arrival never comes (asserted)
r(t) = r_max·sin t/√(E²cos²t+sin²t)   the exact free-fall (closed form, RK4-asserted)
period = 2π, every amplitude          isochrony: the trap is perfect (asserted)
a_hover = tanh(ρ) < c²/L              hovering anywhere is CHEAP (bound asserted)
E_static = cosh(ρ) → ∞                leaving is infinitely expensive (asserted)
ΔE = cosh(ρ₂) − cosh(ρ₁)             the fuel bill — retain energy or fall
```

## Why the corners are unreachable

The bubble is the Poincaré disk: hyperbolic space drawn inside a circle. The rim
sits at map radius 1 — you can *see* it, and the box corners beyond it — but the
proper distance to it is `ρ = 2·atanh(r)`, which diverges. The suite asserts both
directions: distance to the rim grows without bound, and any finite journey
`tanh(ρ/2)` lands strictly inside. Equal-distance rings crowd visibly toward the
rim in the scene: each ring is one more unit of real travel, and they never stop
coming. This is confinement with no wall to touch — the AdS/CFT boundary is where
the dual theory lives (research 46), and no massive traveler visits it.

## The three lessons of stage 1

**Coasting returns you.** Cut the engines anywhere and the wisp free-falls on the
exact geodesic `r(t) = r_max·sin t/√(E²cos²t + sin²t)` (closed form derived from
proper-time SHM; asserted against the RK4 integrator). Every orbit — shallow or
enormous — takes exactly 2π (asserted at amplitudes 1.5 and 4.0): AdS is an
isochronous harmonic trap. The bubble is patient; it always hands you back.

**Hovering is cheap.** A static wisp at proper distance ρ needs proper
acceleration `a = tanh(ρ)` — bounded by `c²/L` (asserted). Any real drive can
hold any altitude forever. Being *somewhere* costs almost nothing.

**Leaving is infinitely expensive.** The energy of a unit-mass wisp at rest at ρ
is `cosh(ρ)` — divergent (asserted). The fuel bill to climb, `cosh ρ₂ − cosh ρ₁`,
is flat at first and then a cliff: each further step costs exponentially more.
This is the precise sense in which the wisp "has to growth an retain energy": the
boundary is not forbidden — it is unaffordable, always, by exactly the amount you
don't have.

## The scene: `wisp_box`

One 16-second cycle of the stage-1 life, inside the box (corner brackets — the
simulation boundary, visible and irrelevant):

* **coast** (one exact 2π period) — the wisp sweeps through the center on the
  closed-form geodesic, comet trail behind it, energy reserve full and idle;
* **burn** — the drive lights, exhaust toward the center; the cyan
  proper-distance ledger climbs steadily while the amber map-radius bar
  saturates *below its violet rim line* (the punchline in one bar: progress
  without arrival) and the magenta reserve drains on the cosh cliff;
* **fall** — the reserve empties, the drive cuts, and the geometry collects:
  released from rest, the wisp whips back through the center, fastest where the
  map is loosest — the hyperbolic signature.

## Stage 2 — the body: orbits are free hover

> "The wisp is like the hand of the mind and the body is an engined you growth
> into and hover."

Stage 1 was a bare point with a drive. Stage 2 grows the body — and the body's
first discovery is that the geometry gives hovering away for free, if you move.

Hovering *still* at proper altitude ρ costs continuous thrust `a = tanh ρ`
(bounded, but forever: cut the engine and you fall). But add sideways motion
and let the centrifugal term carry the weight. The effective radial potential
for angular momentum L is

```
V(r) = (1 + r²)(1 + L²/r²)          (r = sinh ρ, metric radial coordinate)
```

and it is stationary — a genuine circular orbit — exactly when

```
L = r²        E = 1 + r² = cosh² ρ        V″ = 8  (always stable)
```

Three exact facts, each test-asserted:

1. **`L = r²` closes the orbit.** Not an approximation — `V′(r₀) = 0`
   identically, and `V″ = 8 > 0` at *every* radius: there is no ISCO in the
   bubble. Every circular orbit, however deep or high, is stable.
2. **`ω = dφ/dt = 1` universally.** The angular velocity of every circular
   orbit is exactly 1 (in units of c/L), independent of radius. AdS is the
   perfect merry-go-round: two bodies orbiting at different altitudes stay
   forever in phase. This is the orbital face of the same isochrony that made
   every stage-1 coast take exactly 2π.
3. **The rent is `E = cosh² ρ`.** Hovering statically costs `cosh ρ` (the
   stage-1 fuel wall); orbiting at the same shell "costs" `cosh² ρ` — but as
   *kinetic energy you keep*, not thrust you burn. Once paid, the engine goes
   silent and the altitude holds itself.

And the equivalence principle closes the loop: released from rest at ρ, the
exact geodesic is SHM in proper time (`r(τ) = r₀ cos τ`), and for short falls
the proper drop is `½ g τ²` with `g = tanh ρ` — the *same* g the hover engine
fights. Checked against the closed form to 0.1%.

## The scene: `wisp_body`

One 16-second cycle of the stage-2 life:

* **grow** (0–4 s) — the hull assembles around the mote, dash by dash, while
  the green growth ledger fills and the energy reserve charges;
* **climb** (4–7 s) — the drive lights and the body climbs the cosh cliff from
  ρ = 0.35 to the chosen hover shell at ρ = 1.5, reserve draining by the exact
  `cosh ρ₂ − cosh ρ₁` bill;
* **hover** (7–11 s) — engines pinned at `a = tanh 1.5 ≈ 0.905` (the amber
  thrust bar sits just under its white "engine max" line), reserve bleeding —
  altitude held, expensively;
* **orbit** (11–16 s) — the body tips sideways onto the `L = r²` orbit: the
  flame cuts, the thrust bar drops to zero, the reserve goes *flat*, and the
  body circles the shell at ω = 1 — in step with a companion mote that has
  been free-orbiting a lower shell all along, because ω = 1 there too.

## Stage 3 — navigation: the cost algebra of getting anywhere

> "For now we can focus on the fact that for you to move forward in the
> simulation you have to use your drives, your engines."

Stage 3 is travel. The body knows how to hold a shell; now it wants a
*different* shell — and the bubble's geometry turns the whole flight plan into
three closed-form facts (all test-asserted):

**The transfer arc is pure hyperbolic algebra.** The ballistic route between
shells ρ₁ and ρ₂ is the geodesic whose apsides *are* the two shells, and its
conserved constants factor perfectly:

```
E = cosh ρ₁ · cosh ρ₂          L = sinh ρ₁ · sinh ρ₂
```

(asserted: the effective potential equals E² at both turning radii). The whole
arc, exactly: with u = r², radial motion is SHM in u — `u(τ) = ū − A·cos 2τ`,
`ū = (E²−1−L²)/2`, `A = √(ū²−L²)`.

**The fare is path-independent.** Boosting off the ρ₁ orbit costs
`cosh ρ₁ (cosh ρ₂ − cosh ρ₁)`; circularizing at ρ₂ costs
`cosh ρ₂ (cosh ρ₂ − cosh ρ₁)`; the total telescopes to

```
cosh²ρ₂ − cosh²ρ₁  =  orbit_energy(ρ₂) − orbit_energy(ρ₁)
```

exactly the orbit-energy difference (asserted). AdS is conservative: there is
no clever route, no gravity assist, no shortcut — only the fare.

**The subway is isochronous.** Every transfer between *any* two shells takes
coordinate time Δt = π/2 and sweeps Δφ = π/2 exactly (asserted): a quarter
period, a quarter turn, however near or far the destination. The timetable in
the bubble is trivial; only the fare varies. This is the third face of the same
isochrony — free fall (stage 1), orbits (stage 2), and now travel.

**And the geodesic lens.** Release test motes from one point in any direction
with any speed: *all* of them — and the body itself, whose circular orbit is
just one more member of the family — reconverge at the antipodal point at
t = π and come home at t = 2π (asserted to 10⁻³ for three different launches).
In the bubble you cannot get lost; you can only be early with more fuel.

## The scene: `wisp_navigate`

One 16-second cycle of the stage-3 life:

* **orbit A** (0–2.5 s) — riding the ρ = 0.8 shell at ω = 1, reserve full;
* **boost** (2.5 s) — the drive spikes (amber ledger flashes); the magenta
  reserve steps down by the exact boost bill; the dotted route lights up;
* **coast** (2.5–6.5 s) — the quarter-period arc, engines silent, the cyan
  altitude ledger climbing as the body rides pure geometry;
* **circularize** (6.5 s) — the second spike, the second step of the fare —
  which now totals exactly `cosh²ρ_B − cosh²ρ_A`;
* **orbit B** (6.5–9.5 s) — the new shell held for free;
* **the lens** (9.5–16 s) — a fan of test motes is released in every direction:
  they spread, then the bubble folds them all back — antipodal ping (violet
  halo) at t = π, home ping (green halo) at t = 2π, the body arriving in step
  because its own orbit is one more geodesic through the release point.

## Stage 1 in 3D — the ball

> "you are in a sphererical magic circle / bubble"

The brief said *spherical* from the start. The magic circle becomes a magic
sphere — the Poincaré ball, the spatial slice of global AdS₄ — floating inside
the 3D simulation box, corner brackets hanging in depth around it, the camera
orbiting once per cycle. And the first discovery is that *nothing has to
change*: the radial equation never mentioned dimension. The rim is still at
`tanh(ρ/2) < 1`, the isochrony is still 2π, the fuel wall is still `cosh ρ`,
and every free flight is planar (angular momentum is conserved), so the 2D
closed forms apply verbatim in each geodesic's own plane. The wisp's whole
stage-1 life — coast, burn, fall — replays inside the ball, law for law.

What 3D adds is the **size** of the trap, and it is monstrous (all
test-asserted):

* **Areas explode.** The geodesic sphere at proper radius ρ has area
  `A = 4π sinh²ρ` — each shell one unit further out is e² ≈ 7.39× larger
  (asserted). The Euclidean `4πρ²` survives only as the small-ρ limit.
* **The volume is exactly the integral of the area.** `V = π(sinh 2ρ − 2ρ)`,
  with `dV/dρ = A` holding to 10⁻⁸ (asserted), Euclidean `4πρ³/3` at small ρ.
* **The skin theorem.** `V/A → 1/2` as ρ → ∞ (asserted at 10⁻¹⁰): however huge
  the ball grows, essentially *all* of its volume lies within one unit of its
  surface. Hyperbolic space is all skin and no core — the geometric seed of
  holography, visible in a ledger: the bulk lives at its boundary.
* **The isochronous firework.** Release motes from the center in every
  direction with every amplitude: each follows the same closed form along its
  own ray, so the whole swarm passes back through r = 0 *simultaneously* every
  π (asserted). The explosion that un-explodes. The trap is not just perfect
  for one wisp — it is perfect for all of them at once, forever.

## The scenes: `wisp_box_3d` and `wisp_swarm_3d`

`wisp_box_3d` replays the stage-1 cycle in depth: the glowing ball with its
nested equal-ρ shells, the corner brackets of the box hanging in space, the
wisp coasting through the center of the sphere on the exact geodesic, burning
outward to stall just inside the rim (amber ledger pinned under its violet rim
line, magenta reserve draining on the cosh cliff), then falling back when the
fuel wall wins — while the camera orbits the whole trap once per cycle.

`wisp_swarm_3d` is the firework: 40 motes launched from the center in every
direction with every amplitude, spreading through the exponentially growing
shells and then folding back to a single point, twice per cycle. Ledgers: cyan
dispersion breathing; amber shell area (the exponential the motes climb);
magenta the live `2·V/A` rising toward its white ½-asymptote line and never
touching it — the skin theorem, animated.

## Stages 2 and 3 in 3D — the arc, complete in the ball

The body and the trip replay inside the ball with nothing to re-derive — the
laws were dimension-blind all along — and two things become visible that the
flat scenes could only assert:

* **`wisp_body_3d`** — the companion mote now free-orbits a *different plane*
  on its lower shell, and stays in step with the body lap for lap anyway:
  ω = 1 at every radius *in every plane* (the same assert, now seen in depth).
  The hover shell is a full translucent sphere the body holds altitude
  against; the hull grows as a bubble around the core.
* **`wisp_navigate_3d`** — the geodesic lens in its true dimension: the 12
  motes are launched in *different planes* through the release point (every
  geodesic lies in its own plane through the center — angular momentum
  conservation), so the fan blossoms into a genuinely 3D flower — and the ball
  still folds all of it back to the single antipodal point at t = π and home
  at t = 2π. The lens was never a trick of the plane; it is the geometry.

## The boundary sees everything — the wisp's shadow

> "you can never reach those corner **because of AdS/CFT** since you are in a
> sphererical magic circle / bubble"

The brief's *reason* for the trap was holography, and holography has a second
face: the bubble has a boundary theory living on its rim, and the wisp casts an
exact shadow on it. The bulk-to-boundary propagator of global AdS₃, for a
boundary operator of dimension Δ, is

```
K(ρ, θ) = (cosh ρ − sinh ρ · cos θ)^(−Δ)
```

with θ the boundary angle measured from the wisp's direction. Three exact laws
follow, each test-asserted:

1. **The contrast law.** The shadow's peak-to-antipode contrast is `e^{2Δρ}`
   *exactly* (asserted at machine precision, from `cosh ρ ± sinh ρ = e^{±ρ}`).
   The higher the wisp climbs, the more sharply the boundary knows where it
   is — exponentially.
2. **The width law — UV/IR made quantitative.** The half-max angular width has
   the closed form `θ_½ = acos[(cosh ρ − 2^{1/Δ}e^{−ρ})/sinh ρ]` (asserted
   against the numeric half-max), shrinking as `2√(2^{1/Δ}−1)·e^{−ρ}`
   (asserted: `θ_½·e^ρ → 2` for Δ = 1). Bulk depth *is* boundary resolution:
   a deeper wisp is a finer boundary feature.
3. **The conserved imprint.** For Δ = 1, the total shadow `∫K dθ = 2π` at
   *every* ρ (asserted at 10⁻⁶ — exactly, since `cosh²ρ − sinh²ρ = 1`).
   Climbing concentrates the imprint; it cannot change its total. The boundary
   never loses track of the wisp — holography is not surveillance added to
   the bubble; it *is* the bubble.

## The scene: `wisp_shadow`

The stage-1 cycle replayed with its boundary imprint live on the rim: coasting,
the shadow sloshes around the whole circle — uniform at the instant the wisp
crosses the center (ρ = 0 lights the entire boundary at once), then gathering
over the near side; burning, it sharpens into a spike riding just over the
stalled wisp; falling, it relaxes. The cyan ledger is the closed-form width
(shrinks), the amber ledger the log-contrast `2Δρ` (grows) — and the magenta
ledger is the *live numerical integral* of the shadow, pinned under its white
conservation line and never moving.

## The stages, complete

* **Stage 1 — the drives** *(built)*: coast, burn, fall — trapped by geometry,
  the fuel wall named.
* **Stage 2 — the body** *(built)*: the wisp grows into an engine it hovers
  with — hover shells, the `tanh ρ` thrust budget, the `cosh ρ` rent, and the
  orbital discovery that motion makes hovering free.
* **Stage 3 — navigation** *(built)*: geodesic transfers between hover shells
  on hyperbolic algebra, the path-independent fare, the π/2 subway, and the
  lens that makes getting lost impossible.

## Sources

- Operator brief (verbatim above) — the seed of this round
- S. J. Avis, C. J. Isham, D. Storey, *Quantum field theory in anti-de Sitter
  space-time*, PRD 18, 3565 (1978) — AdS as a box: the boundary conditions
- Research 46 (`46-ads-cft-holography.md`) — this repo's AdS/CFT dictionary; the
  Poincaré disk machinery the bubble reuses
- I. Bengtsson, *Anti-de Sitter space* (lecture notes) — global coordinates,
  geodesic isochrony, the cosh potential
- L. Susskind, J. Lindesay, *An Introduction to Black Holes, Information and the
  String Theory Revolution* (2005), ch. on AdS — "AdS is a box" made precise
