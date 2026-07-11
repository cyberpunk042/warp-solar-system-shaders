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

## Presets & scenes

`presets.get(name)` / `presets.names()`: `first`, `binary`, `trinary`,
`blackhole`, `nebula_cradle`, `collapse`. Scenes: `solar_system`, `ss_binary`,
`ss_trinary`, `ss_blackhole`, `ss_nebula`, `ss_collapse`, `ss_flyby` — render with
`python render.py --scene ss_blackhole -o out.png` (add `--frames` to animate).
