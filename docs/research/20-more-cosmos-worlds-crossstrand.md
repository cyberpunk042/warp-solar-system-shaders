# 20 · More cosmic events, new worlds, and the strands crossing

A broad arc that pushes three fronts at once — more violent **cosmic events**, new
**worlds**, and scenes where the engine's separate strands (cosmos · super-earth ·
buildings · life) finally **cross**. Everything reuses machinery already in the
engine; almost no new physics.

## Cosmic events

### Supernova shockwave
A massive star's core collapses and rebounds, driving a shock that unbinds the
star. The ejecta expands at ~10⁴ km/s into a **shock-heated shell** that cools by
blackbody radiation from blue-white through orange to red as it thins — the SN
1987A / Crab-remnant look, with Rayleigh–Taylor **filaments** where the dense
ejecta plows into the shock (Chevalier 1977; Gamezo et al. 2005). We drive the
existing supernova envelope (`stellar_evolution._env_at`/`_march_env`,
`ENV_SUPERNOVA`) with a **growing radius** and an initial **flash**.

### Kilonova (neutron-star merger)
Two neutron stars inspiral (losing energy to gravitational waves), touch, and
merge — the event **GW170817** (LIGO/Virgo + electromagnetic follow-up, 2017). The
merger ejects neutron-rich matter in which **r-process** nucleosynthesis forges the
heavy elements; its radioactive decay lights a **kilonova**: a fast **blue** polar
component (light lanthanide-poor ejecta) and a slower **red** equatorial one
(lanthanide-rich), plus a collimated **short gamma-ray-burst jet**
(Li & Paczyński 1998; Metzger 2019). We render the inspiral, a merge flash, then
the expanding two-colour ejecta cloud + a polar jet.

### Gravitational waves
An inspiralling compact binary radiates **gravitational waves** — quadrupole
ripples in spacetime whose frequency and amplitude **chirp** upward as the orbit
tightens (Einstein 1916; the chirp confirmed by LIGO in 2015). We visualise the
transverse strain as concentric ripples that distort the background starfield,
chirping as the two bodies spiral together.

## New worlds

- **Ringed planet** — a gas/ice world girdled by a **ring system**: the planet
  casts a shadow across the rings and the rings drop a thin shadow band on the
  planet (as on Saturn). Reuses the `gas_giant` ring model + `superearth` body.
- **Ocean moon** — a global-**ocean** world (waves, Fresnel sky-reflection, ice
  caps) with its **gas-giant parent** hanging in the sky — an Europa/exo-moon.
- **Transit / eclipse** — a planet crosses the face of its star (an exoplanet
  **transit**, with the tiny **light-curve** dip that is how most exoplanets are
  found), or a moon eclipses it. Animated.

## The strands crossing

- **City on a planet** — the [buildings](17-buildings.md) SDF city standing on a
  world's surface under a starry sky and a moon: the maintenance/architecture
  strand meets the cosmos strand.
- **Life under twin suns** — the [L-System ecosystem](04-lsystems.md) grown and lit
  by the **Kepler-16 binary** (a warm + a cool sun), casting two coloured shadows —
  the life strand meets the cosmos strand.

## Reuse (not reinvention)

| Piece | From |
|---|---|
| expanding-shell emission integral | `cosmos.stellar_evolution._march_env` |
| blackbody colour | `engine.color.kelvin_to_rgb` |
| binary inspiral / N-body | `cosmos.dynamics`, `cosmos.orbits` |
| planet body + ocean + rings | `superearth`, `earthgfx`, `gas_giant` |
| SDF city | `buildings` |
| lit ecosystem under real suns | `life.render.render_world` |
| starfield + nebula backgrounds | `earthgfx.stars`, `procedural.noise` |

## References

- R. Chevalier, *Interaction of supernovae with circumstellar matter*, ARA&A (1977).
- Li & Paczyński, ApJ 507 (1998); B. Metzger, *Kilonovae*, Living Rev. Rel. (2019).
- LIGO/Virgo, *GW170817* (2017) and *GW150914* (2016).
- Research [15](15-nuclear-fireball.md), [17](17-buildings.md), [19](19-extraordinary-cosmos.md).
