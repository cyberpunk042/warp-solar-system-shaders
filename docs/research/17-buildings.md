# Research 17 — buildings: architecture from signed distance fields

Design behind `warp_shaders/buildings/` — a small library that generates **many
variants of buildings** as signed distance fields, sphere-traced by the same
engine that draws everything else. Buildings are the next *subject* for the
engine, and — deliberately — the next **target**: once they exist as SDFs they
slot straight into the blast renderer's per-cell instancing, so a later arc can
**test the nuke on a city** instead of a forest.

## Why SDFs for buildings

A building is mostly **boxes**: a body, setbacks, a parapet, a pitched roof, a
grid of windows. Signed-distance modelling makes all of that composable with the
primitives we already have (`procedural.sdf`: `sd_box`, `sd_round_box`,
`op_union`, `op_subtract`, …) and — crucially — lets a *whole city* be evaluated
by one function through **domain repetition**, the trick that makes an infinite
skyline cost the same as one tower.

This is an **original Warp reimplementation** of well-understood techniques, built
from Inigo Quilez's canonical [distance functions](https://iquilezles.org/articles/distfunctions/)
and [domain repetition](https://iquilezles.org/articles/sdfrepetition/). It is
*inspired by* — but does not copy — several ShaderToy studies the operator
shared: kishimisu's **"Elevator to Infinity"** (2023, the floor-repeat → infinite
building idea), dr2's **"Opera Island"** (2018, ornate curved architecture), and
the **"House on the Water"** house study. Those are CC-BY-NC-SA works; the code
here is our own.

## The building kit

### Facade — a window grid by repetition + subtraction

The body is a box. Windows are a **grid of small boxes subtracted** from the
facade. Rather than model each window, we fold the surface into one **repeated
cell** and carve a single window there:

```
w = repeat(surface_xy, cell)             # fold to one window cell
facade = subtract( window_box(w), body ) # carve the recessed pane
```

Floors repeat vertically (`rep` on `y`), columns horizontally (`rep` on the
facade tangent). A **parapet/ledge** every N floors is another repeated box added
back on. This is the kishimisu insight: model **one floor**, repeat it.

### Roofs

- **Flat + parapet** (modern tower): the body minus a slab, plus a thin raised
  rim — `op_subtract` then `op_union`.
- **Pitched** (house): a triangular-prism SDF, or the **intersection of two tilted
  half-spaces** capped by the gable ends.
- **Setback** (art-deco tower): stack a few boxes of decreasing footprint.

### Variants

| `sd_*` | Building |
|---|---|
| `sd_tower` | a modern high-rise: body + window grid + parapet + optional setbacks |
| `sd_house` | a house: box body + **pitched roof** + door + window holes |
| `sd_block` | a low rowhouse / office block, wider than tall, banded windows |

### A city — domain repetition + per-cell variation

`city_de(p)` tiles the ground plane into lots with `rep`, derives a **per-lot id**
(`floor(p/cell)`), and **hashes the id** (`procedural.hash`) into that lot's
building parameters — height, width, variant, window phase — so each cell grows a
*different* building from the same code. Streets are the gaps between lots; a
main avenue is a lane where the height is forced low. This is exactly the
instanced-SDF pattern the blast renderer already uses for its forest, so the two
compose.

## Rendering

The scenes sphere-trace `city_de` / `sd_*` with the engine's raymarch + finite-
difference normal, then shade with GGX PBR (concrete / glass materials keyed off
the window mask), engine **soft shadows + AO**, an atmospheric **sky** + **sun**,
and aerial-perspective **fog** for the distance (which also hides the classic
domain-repetition flicker far away). A **night** variant lets windows glow —
emissive panes hashed on per-cell so a fraction are lit — with the bloom pass
turning them into a skyline of lights.

## Forward — buildings as blast targets

The point of an SDF building kit is that it drops into the existing
`blast.render` machinery: that renderer already instances a per-cell **forest**
with a knock-down + char transform driven by the physics-sized shock front.
Swapping the tree instance for a `sd_building` instance (with a rubble/collapse
transform inside the destruction radius) is a small, well-defined follow-up — so
the Tsar Bomba can later be tested on a city, with the damage rings sized by the
same `blast.physics` laws. This arc builds the targets; a later arc aims at them.

## Cross-references

- [Research 00 — foundations](00-foundations.md): the sphere-trace + SDF normal.
- [Research 15 — nuclear fireball](15-nuclear-fireball.md): the blast these
  buildings are built to (eventually) receive.
- Sources: [iq — distance functions](https://iquilezles.org/articles/distfunctions/),
  [iq — domain repetition](https://iquilezles.org/articles/sdfrepetition/).
  Inspiration (not copied; CC-BY-NC-SA): kishimisu "Elevator to Infinity",
  dr2 "Opera Island", "House on the Water".
