# `warp_shaders.cosmos`

The configurable solar system — 1–7 stars (sun / neutron star / white dwarf /
black hole) and configurable planets on chosen orbits, an optional nebula, and a
stable (Kepler) or destructive (N-body) scenario. See
[Research 10](../research/10-solar-system.md) for the physics.

```python
import warp as wp
from warp_shaders.cosmos import presets, render_system
wp.init()

sc = presets.get("first")                 # neutron star + a planet on an ellipse
img = render_system(sc, 960, 540, time=0.0)
```

## Bodies — `cosmos.bodies`

`StarConfig` is a `@wp.struct`; `make_star(kind, radius, temp, activity, spin,
precess, seed)` builds one. Kinds: `SUN`, `NEUTRON`, `WHITE_DWARF`, `BLACK_HOLE`.

| Symbol | Kind | Purpose |
|---|---|---|
| `shade_body(dir, rd, cfg, time)` | device | emissive surface colour (sun granulation / neutron / white dwarf) |
| `body_corona(d_over_r, cfg, time)` | device | corona/glow halo for a ray at `d_over_r`×radius |
| `pulsar_beams(ro, rd, axis, cfg, time)` | device | twin polar beams of a neutron star |
| `render_star(cfg, w, h, …)` | host | render one centred star (verification) |

## Black hole — `cosmos.blackhole`

| Symbol | Kind | Purpose |
|---|---|---|
| `bh_pixel(ro, rd, rs, time, spin)` | device | march the photon-orbit ODE; disk + capture (w<0) |
| `bh_escape_dir(ro, rd, rs)` | device | the lensed exit direction (background sampling) |
| `render_black_hole(cfg, w, h, …)` | host | render one centred hole (lensed starfield + disk) |

## Extraordinary phenomena — `cosmos.{wormhole,quasar,tde}`

Three of the most extraordinary objects and events, all reusing the same GR
photon-bending machinery. See [Research 19](../research/19-extraordinary-cosmos.md).

| Symbol | Kind | Purpose |
|---|---|---|
| `wormhole.render_wormhole(w, h, t, …)` | host | an Ellis throat — this universe lensed into an Einstein ring, another universe fish-eyed through the portal, exotic-matter rim |
| `quasar.render_quasar(w, h, t, …)` | host | a supermassive black hole + Doppler disk + twin **relativistic jets** (synchrotron, drifting shock knots, Doppler-beamed) |
| `quasar.quasar_pixel` / `tde.tde_pixel` | device | photon integrators that reuse `bh_escape_dir` + `_disk_emission` and add jets / a debris stream |
| `tde.render_tde(w, h, t, …)` | host | a **tidal disruption event** — a star spaghettified into a hot log-spiral debris stream + brightening accretion flare |

Scenes: [`wormhole`](../gallery.md), [`quasar`](../gallery.md),
[`tidal_disruption`](../gallery.md).

## More cosmic events — `cosmos.{supernova,kilonova,gwaves}`

More of the universe's most violent events, reusing the stellar-evolution
expanding-shell integrator and the shared starfield. See
[Research 20](../research/20-more-cosmos-worlds-crossstrand.md).

| Symbol | Kind | Purpose |
|---|---|---|
| `supernova.render_supernova(w, h, t, …)` | host | a core-collapse **supernova** — a flash then a self-similar expanding, cooling shock shell (reuses `stellar_evolution._march_env`, ENV_SUPERNOVA) |
| `kilonova.render_kilonova(w, h, t, …)` | host | a **neutron-star merger** — inspiral, merge flash, blue-polar + red-equatorial r-process ejecta and a short-GRB jet |
| `gwaves.render_gwaves(w, h, t, …)` | host | a chirping binary inspiral whose m=2 quadrupole **gravitational-wave** ripples warp the starfield until the pair merges |

Scenes: [`supernova`](../gallery.md), [`kilonova`](../gallery.md),
[`gravitational_waves`](../gallery.md). The worlds (`ringed_planet`,
`ocean_moon`, `transit`) and cross-strand scenes (`city_planet`, `exomoon_life`)
from the same arc are self-contained under `warp_shaders/scenes/`.

## Orbits — `cosmos.orbits`

| Symbol | Purpose |
|---|---|
| `Orbit(a, e, incl, node, arg, period, phase)` | Keplerian elements (`period<=0` pins a body at the focus) |
| `orbit_position(orb, time)` / `orbit_velocity(orb, time)` | world position / velocity (Newton-solved Kepler equation) |
| `solve_kepler(M, e)` · `circular_speed(m, r)` | Kepler solver · circular-orbit speed |
| `nbody_step(pos, vel, mass, dt)` | velocity-Verlet N-body step (mutual gravity) |
| `remnant_type(mass)` · `is_collapse(m0, m1)` | merger physics: mass → star / neutron / black hole + collapse detection |

## Nebula — `cosmos.nebula`

| Symbol | Kind | Purpose |
|---|---|---|
| `nebula_at(p, center, radius, seed, time)` | device | emission colour + density at a point |
| `nebula_march(ro, rd, center, radius, seed, time, steps)` | device | front-to-back integration (emission + transmittance) |
| `render_nebula(center, radius, seed, …)` | host | render one positioned nebula |

## System — `cosmos.system` / `cosmos.dynamics`

| Symbol | Purpose |
|---|---|
| `Star(cfg, orbit)` · `Planet(cfg, orbit, radius)` · `Nebula(center, radius, seed)` | system elements (planet `cfg` is a super-earth `PlanetConfig`) |
| `SystemConfig(stars, planets, nebula, scenario, dist, az, el, fov, tscale)` | a whole system |
| `render_system(sys, w, h, time, device, positions=None)` | host | render the composited system (stars + planets + nebula + lensing) |
| `dynamics.simulate(sys, frames, dt, …)` | run the destructive N-body scenario → a frame list |

## Stellar life-cycle — `cosmos.stellar_evolution`

One star across its whole life on a normalized timeline (see
[Research 11](../research/11-stellar-evolution.md)).

| Symbol | Where | Purpose |
|---|---|---|
| `phase_state(t, mass)` | host | the star's appearance at time `t in [0,1]` for an initial mass: kind / radius / temperature / activity / envelope / flash / H-R coords + phase name |
| `remnant_kind(mass)` | host | the end-state body: white dwarf (`<8 M☉`) / neutron star (`<20`) / black hole |
| `render_lifecycle(t, mass, w, h, …, hr_inset=True)` | host | render the evolving star + envelope (cradle / planetary nebula / supernova ejecta) + the H-R inset; a black-hole remnant delegates to the lensing pass |
| `draw_hr_inset(frame, t, mass, phase)` | host | composite the H-R diagram panel (reference track + trail + marker) into a corner |

Scenes: `stellar_lifecycle` (Sun-like → white dwarf), `stellar_massive`
(14 M☉ → neutron star), `stellar_collapse` (30 M☉ → black hole). `time` walks the
life over 20 s: `python render.py --scene stellar_lifecycle --frames 120 --fps 6 --video life.mp4`.

## Colliding galaxies — `cosmos.galaxy_dynamics`

A Toomre restricted N-body encounter (host, NumPy) — two point-mass cores under
mutual softened gravity, each ringed by massless test particles (see
[Research 12](../research/12-galaxy-collisions.md)).

| Symbol | Where | Purpose |
|---|---|---|
| `GalaxyConfig(mass, n, r_in, r_out, incl_deg, spin, center, vel, color)` | — | one galaxy: core + test-particle disk (spin ±1 = prograde/retrograde) |
| `EncounterConfig(g0, g1, soft, seed)` | — | the two galaxies + gravitational softening |
| `simulate(enc, frames, substeps, dt)` | host | velocity-Verlet integrate the fly-by → a `Collision` (per-frame particle + core positions, galaxy id, colour) |
| `render_collision(sim, frame, w, h, dist, az, el, fov, …)` | host | project + additively splat the star clouds + cores over a starfield, bloomed into galaxy haze |

Scenes: `galaxy_collision` (prograde — long tidal tails + a bridge) and
`galaxy_retrograde` (the same encounter, retrograde — barely any tails). `time`
walks the fly-by over 12 s: `python render.py --scene galaxy_collision --frames 64 --fps 6 --video out/tails.mp4`.

## Presets & scenes

`presets.get(name)` / `presets.names()`: `first`, `binary`, `trinary`,
`blackhole`, `nebula_cradle`, `collapse`. Scenes: `solar_system`, `ss_binary`,
`ss_trinary`, `ss_blackhole`, `ss_nebula`, `ss_collapse`, `ss_flyby` — render with
`python render.py --scene ss_blackhole -o out.png` (add `--frames` to animate).
