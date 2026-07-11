# `warp_shaders.life`

Growing things — L-Systems and the pipeline that turns a grammar into a
ray-traced plant. The pure grammar/geometry layers have no Warp dependency; the
renderer does. See [Research 04 — L-Systems](../research/04-lsystems.md) for the
theory and sources (ABOP).

```python
import warp as wp
from warp_shaders.life import plants
from warp_shaders.life.render import render_plant
wp.init()

spec = plants.get_spec("tree")
mesh, (lo, hi) = plants.grow_mesh(spec, gen=7)      # grow 7 generations
img = render_plant(mesh, 640, 640, eye=(4, 2, 6), target=(0, 1.5, 0))
```

## Core — `life.lsystem`

| Symbol | Kind | Purpose |
|---|---|---|
| `Module(sym, params)` | class | one symbol + numeric parameters |
| `parse(s)` / `word_to_str(w)` | fn | string ↔ list of modules |
| `Rule(pred, produce, left, right, cond, weight)` | class | one production; gates cover all four classes |
| `LSystem(axiom, rules, ignore, seed)` | class | grammar; `.derive(n)` rewrites n generations |

`rules` maps a symbol to a **D0L** string (`"FF"`), a **stochastic** list
(`[("FF", .5), ("F", .5)]`), or `Rule` objects for **context-sensitive**
(`left`/`right`) and **parametric** (`cond` + arithmetic `produce`) grammars.

## Turtle — `life.turtle`

`interpret(word, TurtleConfig) -> Geometry` walks the word with an H/L/U frame
emitting `Segment`s (tapered branches) and `Leaf`s. Commands: `F f + - & ^ \ / |
[ ] ! ' L` (each with an optional parametric argument).

**Environmental response (ABOP §2.3.4).** `TurtleConfig` also carries a *tropism*
layer applied after each drawn segment — the "obvious rules" before any mind:

| Field | Effect |
|---|---|
| `tropism`, `tropism_e` | bend H toward a fixed direction (e.g. gravity `(0,-1,0)`) with susceptibility `e` — **gravitropism / sag** |
| `light`, `light_e` | bend H toward `normalize(light - pos)` recomputed each step — **phototropism** (tracks a moving light) |
| `leaf_fold` (0..1) | pitch leaves down + shrink them — **nyctinasty** (fold shut in rain / at night) |

The bend is `angle = e·|H×T|` about axis `H×T`, so it vanishes when aligned and is
strongest when perpendicular. There is no decision here — a future mind layer sets
these fields to *steer* the plant.

## Mesh — `life.mesh`

`build_mesh(geo, sides=6) -> Mesh` tessellates the geometry into one indexed
triangle mesh (`verts` / `indices` / per-vertex `normals` / `colors`) — tapered
tubes for branches, blades for leaves.

## Render — `life.render`

`render_plant(mesh, width, height, eye, target, sun_dir=…, device="cpu", fov=38,
exposure=1.05, ground_y=0.0) -> (H, W, 3)`. Uploads the mesh as a `wp.Mesh`
(BVH), ray-casts per pixel with `wp.mesh_query_ray`, interpolates the vertex
normal/colour, and shades with GGX PBR + sun + sky + a shadow-catching ground +
the post pipeline.

## Plants — `life.plants`

`get_spec(name)` → a memoized `PlantSpec` for `"grass"`, `"herb"`, `"tree"`,
`"fern"`, `"sapling"`, or `"weeper"`; `grow_mesh(spec, gen) -> (Mesh, (lo, hi))` derives +
tessellates to a generation (cached). The `grass` / `herb` / `tree` **scenes**
grow these with `time`.

`grow_mesh_env(spec, gen, cfg) -> (Mesh, (lo, hi))` interprets a cached word with
an **environment-modified** `TurtleConfig` every call (uncached), so a moving
light or rising rain re-shapes the same structure per frame — the mechanism
behind the `phototropism` / `weeping` / `rain_fold` scenes.
