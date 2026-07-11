# `warp_shaders.procedural`

The building blocks — noise, hashes, and signed-distance functions. Every symbol
here is a **device function** (`@wp.func`): call it inside your own
`@wp.kernel`. Import from the namespace:

```python
from warp_shaders.procedural import fbm3, ridged3, curl3, sd_sphere, op_smooth_union
# or
import warp_shaders as ws
d = ws.procedural.sd_box(...)   # inside a kernel
```

## Noise

All noise takes a `wp.vec3` position and returns a `float` unless noted.
Generators are seamless in space and time where the algorithm allows.

| Function | Signature | Range | Source |
|---|---|---|---|
| `value3` | `(p: vec3) -> float` | `[0, 1]` | value noise (IQ) |
| `noised3` | `(p: vec3) -> vec4` | value in `.w`, **analytic gradient** in `.xyz` | IQ `noised` |
| `perlin3` | `(p: vec3) -> float` | `~[-1, 1]` | Perlin gradient noise |
| `simplex3` | `(p: vec3) -> float` | `~[-1, 1]` | Gustavson/Ashima simplex |
| `value_tiled3` | `(p: vec3, period: float) -> float` | `[0, 1]`, **seamless** | tileable value noise |
| `worley3` | `(p: vec3) -> float` | F1 nearest-feature distance | Worley/cellular |
| `worley3_f2` | `(p: vec3) -> vec2` | `(F1, F2)` distances | Worley F1/F2 |
| `fbm3` | `(p: vec3, octaves: int) -> float` | fractal sum of `value3` | — |
| `fbm_perlin3` | `(p: vec3, octaves: int) -> float` | fractal sum of `perlin3` | — |
| `ridged3` | `(p: vec3, octaves: int) -> float` | ridged multifractal (sharp crests) | Musgrave |
| `billow3` | `(p: vec3, octaves: int) -> float` | billowy (rounded lumps) | — |
| `domain_warp3` | `(p: vec3, octaves: int) -> float` | fBm of a warped domain | IQ domain warping |
| `curl3` | `(p: vec3) -> vec3` | divergence-free flow field | Bridson curl noise |

**Analytic derivatives.** `noised3` returns both the value (`.w`) and the exact
analytic gradient (`.xyz`) in one call — use it for cheap, correct normals on a
noise-displaced surface instead of finite differences.

**Tiling.** `value_tiled3(p, period)` repeats exactly on a `period`-sized
lattice, which is what makes a baked noise texture seamless (see
[textures](textures.md) and the `nebula`/cloud scenes).

**Octaves** are tier-controlled: pass `active_tier().noise_octaves` (or
`qual.noise_octaves`) so detail scales with quality.

## Hashes

Fast, GPU-friendly hash primitives underlying the noise. Deterministic;
input-in, pseudo-random-out.

| Function | Signature |
|---|---|
| `fract` | `(x: float) -> float` |
| `hash11` | `(x: float) -> float` |
| `hash21` | `(p: vec2) -> float` |
| `hash22` | `(p: vec2) -> vec2` |
| `hash31` | `(p: vec3) -> float` |
| `hash33` | `(p: vec3) -> vec3` |

## Signed-distance functions

Exact signed distances (negative inside). Compose primitives with the boolean
operators to build a raymarch `map()`.

### Primitives

| Function | Signature | Shape |
|---|---|---|
| `sd_sphere` | `(p: vec3, r: float) -> float` | sphere |
| `sd_box` | `(p: vec3, b: vec3) -> float` | box (half-extents `b`) |
| `sd_round_box` | `(p: vec3, b: vec3, r: float) -> float` | box with rounded edges |
| `sd_torus` | `(p: vec3, t: vec2) -> float` | torus (major/minor `t`) |
| `sd_cylinder` | `(p: vec3, h: float, r: float) -> float` | capped cylinder |
| `sd_capsule` | `(p: vec3, a: vec3, b: vec3, r: float) -> float` | capsule between `a` and `b` |
| `sd_plane` | `(p: vec3, n: vec3, h: float) -> float` | plane (normal `n`, offset `h`) |
| `sd_ellipsoid` | `(p: vec3, r: vec3) -> float` | ellipsoid (radii `r`) |

### Operators

| Function | Signature | Effect |
|---|---|---|
| `op_union` | `(a: float, b: float) -> float` | `min(a, b)` |
| `op_intersect` | `(a: float, b: float) -> float` | `max(a, b)` |
| `op_subtract` | `(a: float, b: float) -> float` | `max(-a, b)` |
| `op_smooth_union` | `(a: float, b: float, k: float) -> float` | blended union |
| `op_smooth_subtract` | `(a: float, b: float, k: float) -> float` | blended subtraction |
| `op_smooth_intersect` | `(a: float, b: float, k: float) -> float` | blended intersection |
| `op_round` | `(d: float, r: float) -> float` | inflate a surface by `r` |
| `op_onion` | `(d: float, thickness: float) -> float` | hollow shell |

## Fractal distance estimators

Escape-time 3D fractals have no surface equation but admit a **distance
estimator** — a lower bound on the distance to the set — so the ordinary
sphere-tracer marches them like any SDF. Each returns a `wp.vec4`:
`(de, trap, escape_iter, final_r)`, where `trap` is the **orbit trap**
(minimum `|z|` over the iteration) — the colour signal for the banded shells.

| Function | Signature | Fractal |
|---|---|---|
| `mandelbulb_de` | `(p: vec3, power: float, iters: int) -> vec4` | Mandelbulb (White–Nylander triplex power; `power=8` is the classic) |
| `mandelbox_de` | `(p: vec3, scale: float, iters: int) -> vec4` | Mandelbox (Lowe box-fold + sphere-fold; `scale=-1.5` / `2` are classics) |

The analytic Mandelbulb estimator `0.5·log(r)·r/dr` deliberately over-reports
far from the surface (only near-surface accuracy matters); march with a fudge
factor `< 1`. See the [`mandelbulb`](../gallery.md) and [`mandelbox`](../gallery.md)
scenes and [Research 13](../research/13-3d-fractals.md).

> **Sources.** SDF primitives and operators follow Inigo Quilez's distance-
> function reference; noise follows IQ, Stefan Gustavson, Steven Worley, and
> Robert Bridson. See [`docs/research/`](../research/00-foundations.md) for the
> citations.
