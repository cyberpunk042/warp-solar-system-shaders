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

## The stages ahead (named, not built — "we do not want to overdo it")

* **Stage 2 — the body**: the wisp grows into an engine it hovers with:
  persistent structure, hover shells at chosen ρ (each shell's `tanh ρ` thrust
  budget and `cosh ρ` rent), energy harvesting and retention between burns.
* **Stage 3 — navigation**: moving forward through the simulation on drives:
  hyperbolic waypoints, geodesic transfers between hover shells, the cost
  algebra of getting anywhere in a space that resists arrival.

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
