# `warp_shaders.superearth`

A configurable procedural planet — one Warp kernel driven by a `PlanetConfig`
struct. Every feature is an independent knob, so the same code renders a barren
rock, a living earth, an ocean world, a gas giant, or a world under nuclear
bombardment. See [Research 09 — Super-Earth](../research/09-super-earth.md) for
the techniques and sources.

```python
import warp as wp
from warp_shaders.superearth import presets, render_planet, make_config
wp.init()

cfg = presets.get("earthlike")                    # or make_config(mountain=0.0, …)
img = render_planet(cfg, 960, 540, time=0.0, quality="high")
```

## Config — `superearth.planet`

`PlanetConfig` is a `@wp.struct`; build it with `make_config(**kw)` (any field
overridable by name, sensible defaults for the rest).

| Field | Meaning |
|---|---|
| `seed` | procedural seed (varies continents, volcanoes, storms) |
| `mountain` | mountain amplitude — **0 = flat continents** |
| `sea_level` | ocean threshold in elevation units |
| `has_ocean` / `has_lakes` / `has_rivers` | water bodies (int flags) |
| `snow` | snow-cap amount (poles + genuine peaks) |
| `has_volcano` / `volcano_n` / `lava` | volcanism + molten glow |
| `veg` | vegetation amount |
| `alive` / `city` | bioluminescence + night-side city lights |
| `has_atmo` / `atmo` / `cloud` | atmosphere density + cloud coverage |
| `gas` / `storm` / `electro` | super-planet: banded gas / windstorm / electrostorm |
| `spin` | rotation rate |

| Function | Kind | Purpose |
|---|---|---|
| `make_config(**kw)` | host | build a `PlanetConfig` with defaults + overrides |
| `render_planet(cfg, w, h, time, mouse, device, quality, sun_az, sun_el, dist, fov, moons, relief)` | host | render `cfg` (and any moons) to an `(H, W, 3)` image; `relief=False` uses the fast analytic surface (no per-pixel march) |

## Presets — `superearth.presets`

`presets.get(name)` returns a config; `presets.names()` lists them:
`barren`, `earthlike`, `arid`, `ocean_world`, `volcanic`, `riverlands`,
`living`, `flatland` (mountains off), `gas_giant`, `windstorm`, `electrostorm`.

## Moons — `superearth.moons`

| Symbol | Kind | Purpose |
|---|---|---|
| `Moon(orbit, size, speed, phase, incl, kind)` | class | one moon on an inclined circular orbit (kind: rocky/icy/lava/desert) |
| `moon_state(moons, time)` | host | → `(positions Nx3, radii N, type-ids N)` at `time` |
| `moonset(name)` | host | a named system — `none` / `luna` / `twin` / `many` |

## Bombardment — `superearth.bombardment`

| Symbol | Kind | Purpose |
|---|---|---|
| `BombConfig(n, delay, interval, parallel, formula, yield_scale, seed)` | class | warhead count / timing / distribution knobs |
| `sites(n, formula, seed, front)` | host | `n` unit strike vectors (`uniform`/`clustered`/`equatorial`/`spiral`), optionally front-biased |
| `run(planet_cfg, bcfg, w, h, frames, dt, device, dist, fov, moons, quality)` | host | render the full sequence → list of composited frames |

## Scenes

`super_earth`, `se_barren`, `se_ocean`, `se_volcanic`, `se_rivers`, `se_living`,
`se_arid`, `se_flat`, `se_moons`, `se_gas`, `se_windstorm`, `se_electrostorm`,
and `se_nuked` (the bombardment). Render any with
`python render.py --scene se_gas -o out.png`.
