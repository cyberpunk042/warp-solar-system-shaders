# `warp_shaders.buildings`

Architecture as **signed distance fields** — a parametric kit of buildings that
sphere-trace like any SDF, plus whole cities and suburbs from one function via
domain repetition. An original Warp reimplementation of standard SDF-architecture
techniques (Quilez [distance functions](https://iquilezles.org/articles/distfunctions/)
+ [domain repetition](https://iquilezles.org/articles/sdfrepetition/)); inspired
by — not copied from — ShaderToy studies by kishimisu and dr2. See
[Research 17](../research/17-buildings.md).

## Building primitives (`@wp.func`)

| Function | Building |
|---|---|
| `sd_tower(p, half, floor_h, win_w)` | modern high-rise: body + protruding floor bands + corner pilasters + parapet + base |
| `sd_house(p, half, roof_h)` | house: box body + pitched (gable) roof + carved door/windows |
| `sd_block(p, half, floor_h)` | low, wide office block with a roof cap |
| `sd_triprism(p, hx, hz)` | a triangular prism (the gable roof) |

`half` is the body half-extents about its centre. All are **clean solids** — the
window grid is a shading detail, not carved geometry (a carved lattice would
tunnel through and show holes).

## Cities from one function

| Function | Returns |
|---|---|
| `city_de(p, lot, seed)` | `(dist, height, variant, lot_rand)` — a downtown of towers/blocks |
| `suburb_de(p, lot, seed)` | `(dist, body_half_h, variant, lot_rand)` — a neighbourhood of houses |

Each tiles the ground into `lot`-sized cells, hashes the cell id into that lot's
building parameters (height / footprint / variant), and grows a different building
per cell — streets are the gaps. Keep the footprint well inside the lot: buildings
that reach the cell wall break the distance field (sphere-tracing overshoots), so
march with a **small step** (`t += d * 0.6`) and give buildings room.

## Windows

`window_mask(p, cell_xy, floor_h)` returns 1 on a glass pane, 0 on the mullions —
use it to pick glass vs concrete, and (hashing a per-window id) which panes glow.
The [`city`](../gallery.md) scene lights ~half the panes as warm emissive windows
and blooms them into a night skyline; [`suburb`](../gallery.md) shades plaster
walls + terracotta roofs under a warm sun.

## Forward — blast targets

Because they are clean instanced SDFs, these drop into `blast.render`'s per-cell
instancing: a later arc can swap the forest for buildings and **test the nuke on a
city**, with the damage rings sized by `blast.physics`.
