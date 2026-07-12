# 18 · Testing the nuke on a city

The [buildings](17-buildings.md) arc built an SDF architecture kit *"to later test
the nuke on."* This is that test: the [nuclear detonation](15-nuclear-fireball.md)
rendered over a **city** instead of a forest, with the buildings collapsing to
rubble as the blast wave sweeps out. It fuses two existing strands — the
`buildings` SDF kit and `blast.physics` — into one scene, and nothing here is new
physics; it is a new *target* for the physics we already have.

## Overpressure damages structures in rings

A nuclear air-burst destroys by **peak static overpressure** (the crushing jump in
air pressure behind the shock front), and structural damage is tabulated against it
(Glasstone & Dolan, *The Effects of Nuclear Weapons*, 1977, ch. V). The canonical
contours the engine already sizes in `blast.physics`:

| Overpressure | Effect on structures | `physics` radius | 300 kt | 50 Mt |
|---|---|---|---|---|
| **20 psi** | Reinforced-concrete buildings destroyed | `severe_radius` | 1.9 km | 10 km |
| **5 psi** | Most residential / commercial buildings collapse — *total destruction* | `destruction_radius` | 6.9 km | 38 km |
| **1 psi** | Window breakage, light injuries | `light_radius` | 20 km | 108 km |

Damage grades with distance: inside the 20 psi ring even hardened structures are
flattened; out to the 5 psi ring ordinary buildings collapse; between 5 and 1 psi
they are damaged but stand. That gradient is exactly what the render draws — a city
that is **rubble at the centre, cracked-but-standing at the edge, intact in the far
suburbs** — so the same frame shows the whole overpressure ladder at once.

## The collapse model

Each lot's building (from `buildings.city_de`'s domain-repetition scheme) gets a
**collapse factor** `c ∈ [0,1]` from its distance to ground zero:

- the overpressure **front** expands to the final 5 psi radius over the shot (the
  visualization compresses the few real seconds of wave propagation — the front is
  the moving `destruction_radius` boundary);
- a building the front has passed collapses, *harder* the closer it is to the 20 psi
  `severe_radius` (deeper in ⇒ nearer total);
- collapse **crushes the building height** toward the ground and **piles a rubble
  mound** in its footprint, and **chars** its material (scorch grades with `c`).

So over `--frames` the wave eats the skyline outward from the fireball: towers near
ground zero drop to charred rubble first, the ring of destruction widens, and the
mushroom climbs above a flattened downtown — the standing physics, now with a city
underneath it.

## What is reused (not reinvented)

| Piece | From |
|---|---|
| `sd_tower` / `sd_block` + domain repetition | `buildings.sdf` (arc 17) |
| fireball / mushroom / condensation ring / blackbody cooling | `blast.render` + `blast.physics` (arc 15) |
| 20 / 5 / 1 psi damage radii | `blast.physics.{severe,destruction,light}_radius` |
| shock-ring structure | `physics.shock_ring` (ported from `the-virus-block-mc`) |

The only new code is the **collapse SDF** (`_city_blast_de`) that reduces a
building's height + adds rubble by its overpressure grade, and a city-shading render
path — everything else is the two libraries meeting.

## Ethical note

This is a visualization of **published civil-defence physics** — the Glasstone &
Dolan overpressure-damage table is textbook material whose purpose is to convey the
scale of these weapons. It is a rendering of consequences, not weapon-design data,
in the same spirit as the [fireball research](15-nuclear-fireball.md).

## References

- Glasstone & Dolan, *The Effects of Nuclear Weapons*, 3rd ed., 1977 — ch. V (air
  blast; the overpressure–damage table).
- Quilez, [distance functions](https://iquilezles.org/articles/distfunctions/) +
  [domain repetition](https://iquilezles.org/articles/sdfrepetition/) — the SDF city.
- Research [15 · Nuclear fireball](15-nuclear-fireball.md) and
  [17 · Buildings](17-buildings.md) — the two strands this fuses.
