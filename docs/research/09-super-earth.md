# Research 09 — Super-Earth: one planet, every knob

Sources and reasoning behind `warp_shaders/superearth/` — a single Warp kernel
that renders a whole family of worlds from one config struct: rocky earths with
or without oceans, lakes, rivers, mountains, snow, volcanoes, lava, vegetation,
life and cities; gas super-planets (banded, windstorm, electrostorm); a
configurable moon system; and a configurable orbital bombardment.

## The design: config *is* the planet

Every feature is a field on a `@wp.struct PlanetConfig` — floats and ints, so it
rides straight into the kernel with no branching cost worth worrying about. The
kernel reads the struct and decides, per pixel, what the surface is. Turning a
feature on or off is a config change, never a code change. This is the same
"structured data drives behaviour" idea the L-System layer uses for grammars,
applied to a planet: the world is *declared*, not hand-built.

`make_config(**kw)` fills sensible defaults and overrides any field by name;
`presets.py` names useful points in that space (earthlike, barren, arid,
ocean_world, volcanic, riverlands, living, flatland, gas_giant, windstorm,
electrostorm).

## The surface: a displaced sphere

The rocky worlds are a **heightfield-displaced sphere**, ray-marched per pixel so
the relief is real on the limb. Elevation is a sum of procedural layers on the
unit direction `d` (all standard techniques from Ken Perlin's gradient noise and
Steven Worley's cellular noise):

- **continents** — domain-warped fBm thresholded into land/sea masks. Domain
  warping (warping the sample point by another noise field, per Inigo Quilez's
  articles on fBm and warping) breaks up the tell-tale roundness of raw noise.
- **mountains** — *ridged* multifractal noise (`1 - |noise|`, stacked), the
  standard way to get sharp crests instead of rolling hills (Musgrave, *Texturing
  & Modeling: A Procedural Approach*). The `mountain` knob is its amplitude — set
  it to 0 and the continents go flat (the `flatland` preset).
- **volcanoes** — a small set of cones with a crater dip, placed on a Fibonacci
  sphere (the golden-angle spiral, the classic even point distribution on a
  sphere).

The rendered radius floods to sea level where terrain drops below it, so oceans
are simply "where the surface is the water shell". The surface normal is the
gradient of that radius field, taken with two tangent offsets — cheap, and it
captures both the terrain slope and the water plane.

**Rivers and lakes** reuse the *second-nearest* Worley feature: `F2 − F1` is
small exactly on the boundaries between cells, which draws a branching channel
network (rivers) and, thresholded differently on fBm basins, isolated water
bodies (lakes). Cellular-noise cell edges as a river/crack network is a
well-known trick from procedural texturing.

## Shading

Land is a small elevation-keyed rock palette (lowland → highland → bare peak)
with fBm mottling; **vegetation** tints moist, temperate, lowland bands green;
**snow** is gated on genuine altitude-above-sea *and* latitude, kept separate
from the colour ramp so raising the land doesn't paint the mid-latitudes white.
Water is Cook-Torrance GGX (the engine's `shade_pbr`) with an ocean-depth tint
and a calmer freshwater material. A **living** world adds bioluminescent vein
networks (Worley edges again) that pulse on the night side, clustered into
"biomes" by a low-frequency mask; **city lights** stipple the night land.
Everything sits under this module's own configurable single-scatter atmosphere
and an optional volumetric cloud shell.

Lava is emissive and overrides the surface where a vent or a young-world molten
sea is hot; the palette and cooled-crust rafts follow the same approach as the
engine's standalone lava-planet scene.

## Super-planets: gas, windstorm, electrostorm

A **gas giant** has no solid surface — the whole disk is atmosphere — so when
`gas > 0` the kernel skips the terrain march entirely and shades a smooth sphere
with banded *zonal flow*. Gas giants organise into alternating light **zones**
and dark **belts** at fixed latitudes (Jupiter's visible structure), driven by
banded jet streams. We model that as `sin(latitude · k)` for crisp horizontal
bands, with the band edges gently displaced by a domain-warp field so they read
as *flowing* turbulence rather than painted stripes — but the displacement is
kept small so the bands never scramble into blobs. A large oval **anticyclone**
(a Great-Red-Spot analogue) is a stretched Gaussian with an internal swirl.

- **windstorm** raises the turbulence and scatters cyclone **eyes** (Worley
  cells) through the bands — a world whipped into chaotic flow.
- **electrostorm** darkens the bands to slate thunderheads and adds a **branching
  lightning web**: the Worley `F2 − F1` cell-edge network again, but gated to a
  few drifting active storm cells and strobed in time, so a handful of blue-white
  branches crackle and move between frames instead of the whole globe lighting at
  once.

## Moons

`moons.py` places any number of moons on inclined circular orbits (radius, size,
speed, phase, inclination, and one of four surface types: rocky, icy, lava,
desert). They live in world space — unrotated by the planet's spin — and the
kernel intersects them as plain spheres, nearest-in-front wins. `moonset()` names
a few systems (none / luna / twin / many).

## Bombardment: configurable orbital strikes

`bombardment.py` rains warheads on a super-earth. The knobs are the operator's:
warhead **count**, spatial **distribution formula** (uniform / clustered like
real arsenals / equatorial / spiral), **delay** before the first strike,
**interval** between waves, and how many detonate **in parallel** per wave. Each
strike is built on the engine's `ParticleSystem`: a hot radial ejecta plume that
cools through a blackbody ramp, plus an expanding **shock-ring scar** on the
surface (white-hot young, cooling to orange). The particles are composited over
the ray-marched globe with a front-hemisphere occlusion cull (a particle splat
has no globe to hide behind, so the far side must be culled by hand), and the
strike sites are biased onto the camera-facing hemisphere so the bombardment
clearly lands on the face we see. The whole sequence is rendered once with the
fast analytic surface path (no per-pixel march) and cached, so `--frames`/`--gif`
plays it back.

## Sources

- Ken Perlin, *An Image Synthesizer* (SIGGRAPH 1985) and *Improving Noise*
  (2002) — gradient noise, the fBm building block.
- Steven Worley, *A Cellular Texture Basis Function* (SIGGRAPH 1996) — Worley /
  Voronoi noise; `F1`, `F2`, and `F2 − F1` cell edges for rivers, veins, and
  lightning.
- F. Kenton Musgrave et al., *Texturing & Modeling: A Procedural Approach* —
  ridged multifractals for mountains, procedural planet construction.
- Inigo Quilez — articles on fBm, domain warping, and raymarching implicit
  surfaces (iquilezles.org).
- The Fibonacci-sphere (golden-angle) point distribution for volcanoes and moon
  seeding.
- Jupiter's zone/belt banding and Great Red Spot as the reference for the gas
  super-planet's zonal-flow shading.
- Cook & Torrance microfacet BRDF (SIGGRAPH 1981), realised in the engine's
  `shade_pbr`, for the water and land lighting.
