# `warp_shaders.sim` — physics simulators

Time-stepped physics on Warp: a state advanced by the governing equations each step, rendered
from whatever the dynamics produce. Every module here is trivially parallel (one thread per body
/ grid cell) and matches the analytic physics; they run on CPU here and scale straight up on CUDA.

See [Research 40](../research/40-physics-sims.md) (gravity, fluids) and
[Research 41](../research/41-waves-and-resonance.md) (waves) for the physics behind them.

## `sim.nbody` — N-body gravity

Softened Newtonian gravity summed pairwise (`O(N²)`) in a Warp kernel, advanced with a
symplectic **leapfrog** integrator.

| Symbol | Purpose |
|---|---|
| `NBody(pos, vel, mass, device, g, eps)` | State container; `.step(dt)` and `.run(steps, dt) -> (pos, vel)`. |
| `make_collision(n, sep, radius, approach, spin, impact, seed)` | Build two Plummer clumps on a (grazing) collision course → `(pos, vel, mass, clump)`. |

Consumed by the `nbody` scene (two star clouds colliding, tidal tails, merged core).

## `sim.fluid` — 2-D incompressible Navier–Stokes (Stable Fluids)

Jos Stam's method: divergence-free **pressure projection** (Jacobi), **semi-Lagrangian**
advection, buoyancy and vorticity confinement.

| Symbol | Purpose |
|---|---|
| `StableFluid(n, buoy, vort, seed)` | Grid state; `.step(dt, phase)`, `.run(steps, dt) -> (d, t, vx, vy)`, `.emit(dt, phase)`. |

Consumed by the `fluid` scene (a rising, curling smoke plume off a hot emitter).

## `sim.wave` — 2-D wave equation

Explicit leapfrog stencil for `u_tt = c²∇²u` with point/line oscillators, an optional barrier
mask, and an absorbing border (stable while the Courant number `c·dt/h ≤ 1`).

| Symbol | Purpose |
|---|---|
| `WaveField(n, c, damp, border)` | Grid state; `.step(t)`, `.run(steps) -> u`, `.laplacian()`. |
| `.add_source(x, y, amp, omega, phase)` | A point oscillator. |
| `.add_line_source(y, amp, omega, x0, x1, phase)` | A row of in-phase oscillators (plane wave). |
| `.double_slit(y, gap, sep, thickness)` | A reflecting barrier with two slits. |

Consumed by the `ripple_tank` and `double_slit` scenes.

## `sim.engine` — particle system + splatting

The general particle substrate (buoyant/drag integration + additive splatting) behind the blast
and impact scenes.

| Symbol | Purpose |
|---|---|
| `ParticleSystem(max_n, device)` | Ring-buffered particles; `.spawn(...)`, `.step(dt, g, buoy, drag, cool, ...)`, `.render(...)`. |
| `splat_points(width, height, pos, col, bright, eye, target, ...)` | Additive-splat coloured points to an `(H, W, 3)` image (used by `nbody`). |

## `sim.blast` — nuclear / thermonuclear blast staging

Stage-based particle spawning (fireball, neutrons, rising column, base surge) for the nuclear
detonation sims. `simulate(scenario, drop, frames, dt, ...)`.

## `sim.earth` — Earth-impact + self-gravity

A volume-filled particle Earth under central-field self-gravity, with a realistic globe renderer.
`build_earth(n, rng)`, `simulate_earth(arsenal, outcome, frames, n, ...)`.

## Related

The **path-tracing** Monte-Carlo helpers (`onb_cosine`, `sample_sphere`, `camera_basis`) live in
[`engine`](engine.md) (`engine.pathtrace`); the electricity toolkit (fractal bolts, point glow,
corona) lives in `warp_shaders.electric`.
