# Gallery

Every scene is one module in `warp_shaders/scenes/`, rendered with
`python render.py --scene NAME --quality high -o out.png`. Run
`python render.py --list` for the full, current list (99 scenes).

## Engine showcase

The hero scenes — each composes the engine (procedural + PBR + atmosphere +
volumetrics + post) and honours `--quality low..ultra`.

| | | |
|---|---|---|
| **Earth v2** (flagship)<br>PBR ocean + real atmosphere + volumetric clouds<br>![earth_v2](engine/earth_v2.png) | **baked-map Earth**<br>drop-in NASA equirect texture + atmosphere<br>![earth_map](engine/earth_map.png) | **sky**<br>Rayleigh + Mie atmospheric scattering<br>![sky](engine/sky.png) |
| **volumetric clouds**<br>HG phase, Beer–Lambert, sun light-march<br>![clouds](engine/clouds.png) | **PBR demo**<br>GGX raymarch, soft shadows, AO, bloom<br>![pbr_demo](engine/pbr_demo.png) | **noise gallery**<br>fBm / Perlin / Worley / ridged / warp / curl<br>![noise](engine/noise_gallery.png) |
| **terrain**<br>raymarched heightfield + aerial perspective<br>![terrain](engine/terrain.png) | **ocean**<br>analytic waves, Fresnel sky, GGX glitter, foam<br>![ocean](engine/ocean.png) | **nebula**<br>emissive volume from a baked 3D noise field<br>![nebula](engine/nebula.png) |
| **gas giant + rings**<br>banded atmosphere, red spot, ring shadows<br>![gas_giant](engine/gas_giant.png) | **alien world**<br>twin coloured suns, violet sky, jagged terrain<br>![alien](engine/alien.png) | **spiral galaxy**<br>log-spiral arms, core bulge, HII knots<br>![galaxy](engine/galaxy.png) |
| **aurora**<br>volumetric light curtains over a night landscape<br>![aurora](engine/aurora.png) | **lava planet**<br>molten sea, cooled-crust rafts, basalt islands<br>![lava_planet](engine/lava_planet.png) | **desert dunes**<br>wind ripples, long low-sun shadows, aerial haze<br>![dunes](engine/dunes.png) |
| **glacier**<br>blue ice + snow, subsurface glow, cold low sun<br>![glacier](engine/glacier.png) | **depth of field**<br>thin-lens focus pull, near/far bokeh<br>![dof_showcase](engine/dof_showcase.png) | **slot canyon**<br>layered sandstone + volumetric god-rays<br>![canyon](engine/canyon.png) |
| **underwater reef**<br>rippling caustics, blue-green depth, god-rays<br>![reef](engine/reef.png) | **post-FX showcase**<br>blackbody orbs + starfield + full post chain<br>![postfx](engine/postfx.png) | **soft shadows + AO**<br>analytic sphere shadows + ambient occlusion, no SDF march<br>![shadow_demo](engine/shadow_demo.png) |
| **reflections**<br>Whitted mirror + glass + gold spheres, reflecting each other (bounce loop)<br>![reflections](engine/reflections.png) | | |

## 3D fractals

Distance-estimated fractals (`warp_shaders.procedural.fractal`), sphere-traced
like any SDF and coloured from the **orbit trap**. Escape-time
([Research 13](research/13-3d-fractals.md)) and folding IFS
([Research 14](research/14-kifs-fractals.md)).

| | | |
|---|---|---|
| **Mandelbulb**<br>White–Nylander triplex power, glowing; power morphs 2→8<br>![mandelbulb](engine/mandelbulb.png) | **Mandelbox**<br>Lowe box-fold + sphere-fold (scale −1.5), the ringed cube<br>![mandelbox](engine/mandelbox.png) | **Menger sponge**<br>Quilez exact SDF, drilled-cube recursion 1→4<br>![menger](engine/menger.png) |
| **Sierpinski tetrahedron**<br>plane folds + scale, crystalline 3D gasket<br>![sierpinski](engine/sierpinski.png) | **kaleidoscopic temple**<br>KIFS fold+rotate+scale — fractal architecture<br>![kifs_temple](engine/kifs_temple.png) | |

## Life — molecule to cell

The bottom of the "show life" ladder (`warp_shaders.life.molecular` /
`.cell`) — DNA and proteins as solid ray-traced meshes, the cell as a soft glow
volume. Each **animates**: the helix assembles, the chain folds, the cell
divides. See [Research 05](research/05-molecular-to-cell.md).

| | | |
|---|---|---|
| **DNA**<br>double helix, colour-coded base pairs (B-DNA)<br>![dna](life/dna.png) | **protein**<br>backbone folding extended→compact, N→C colour<br>![protein](life/protein.png) | **cell**<br>membrane + nucleus + organelles, dividing<br>![cell](life/cell.png) |

Assembling, folding, dividing:

![dna assembling](life/dna_assemble.gif)
![protein folding](life/protein_fold.gif)
![cell dividing](life/cell_divide.gif)

## Life — grown from L-Systems

Real plants grown from L-System grammars (`warp_shaders.life`), tessellated to
a mesh and **ray-cast** through the Warp engine. Generation advances with
`time`, so they grow. See [Research 04](research/04-lsystems.md).

| | | |
|---|---|---|
| **grass**<br>tuft of arching blades<br>![grass](life/grass.png) | **herb**<br>stochastic leafy plant, golden-angle leaves<br>![herb](life/herb.png) | **tree**<br>parametric tapering tree + leafy canopy<br>![tree](life/tree.png) |
| **fern**<br>bracketed frond unfurling into a fiddlehead (ABOP fig 1.24)<br>![fern](life/fern.png) | **flower**<br>leafy stem that blooms into a whorl of petals at maturity<br>![flower](life/flower.png) | **bush**<br>dense stochastic shrub, wide and leafy<br>![bush](life/bush.png) |

Growth (`--frames 8 --fps 1`), sprout → tree, and the fern unfurling:

![tree growing](life/tree_grow.gif)
![fern unfurling](life/fern_grow.gif)

A whole **meadow** — grass, herb, flower, fern and bush merged into one mesh and
swaying in a single wind:

![meadow](life/meadow.png)
![meadow swaying](life/meadow.gif)

### Environmental response (the "obvious rules", ABOP §2.3.4)

Before any mind, the plant obeys physics — a **tropism** bends the turtle's
heading toward a direction each step, so the same grammar reacts to its world.

| | | |
|---|---|---|
| **phototropism**<br>sapling bends to follow a moving light<br>![phototropism](life/phototropism.png) | **weeping**<br>shoots sag under gravity into a fountain<br>![weeping](life/weeping.png) | **rain-fold**<br>leaves fold shut as rain sets in<br>![rain-fold](life/rain_fold.png) |
| **wind**<br>tuft swaying as a gust pulses (time-varying tropism)<br>![wind](life/wind.png) | | |

Following the light (`phototropism`), closing in the rain (`rain_fold`), and
swaying in a gust (`wind`):

![light tracking](life/photo_track.gif)
![rain fold](life/rain_fold.gif)
![wind sway](life/wind.gif)

### The mind — choosing to obey (Conway's Life)

Top of the ladder: a Conway Game-of-Life **mind** whose living population sets a
**drive** that *chooses* whether the plant seeks the light (open, phototropic) or
rests (sags, leaves folded) — a decision, not a reflex. The inset shows the grid
+ drive bar. See [Research 06](research/06-the-mind.md).

![the mind choosing](life/mind.png)
![the mind deliberating](life/mind.gif)

And a **per-branch** mind (`mind_branches`) — each shoot of one plant steered by a
different band of the grid, so some reach for the light and open while others sag
and fold shut, all at once ("close *piece of itself*"):

![per-branch mind](life/mind_branches.png)
![per-branch mind animating](life/mind_branches.gif)

### Wave and collapse — futures resolving to one

The summit of the strand: several *possible* plant futures begin **superposed**
(a faint overlapping ghost cloud of what the plant might become) and **collapse**
to a single realised plant — the front sweeping tip→base (the future settling
first, reaching *backward* into its own history), with the Conway mind biasing
*which* future resolves. See [Research 07](research/07-wave-and-collapse.md).

![wave collapsing to one plant](life/wavecollapse.png)
![wave collapse animating](life/wavecollapse.gif)

### Ecosystem — a living meadow over the seasons

Life at the population scale: a patch of plants that live over **years** — born,
growing, blooming, senescing, dying, reseeding — **recolouring with the seasons**
and **competing for light** (a shaded plant grows less and leans toward the open
sky). See [Research 08](research/08-ecosystem.md).

| summer | autumn | winter |
|---|---|---|
| ![summer meadow](life/ecosystem.png) | ![autumn meadow](life/ecosystem_autumn.png) | ![winter meadow](life/ecosystem_winter.png) |

A few years, seasons cycling and the meadow turning over:

![ecosystem over the years](life/ecosystem.gif)

## Super-Earth — one planet, every knob

One Warp kernel driven by a `PlanetConfig` struct: turn oceans, lakes, rivers,
mountains, snow, volcanoes, lava, vegetation, life, cities, atmosphere, clouds
and moons on or off independently — same code, a whole family of worlds. Then the
**super-planets** (no solid surface) and a configurable orbital bombardment. See
[Research 09](research/09-super-earth.md).

| | | |
|---|---|---|
| **earth-like**<br>oceans, green continents, rivers, snow, clouds<br>![super_earth](superearth/super_earth.png) | **barren**<br>cratered dead rock, no air or water<br>![barren](superearth/se_barren.png) | **ocean world**<br>mostly water with island arcs<br>![ocean](superearth/se_ocean.png) |
| **volcanic**<br>young molten world, lava vents<br>![volcanic](superearth/se_volcanic.png) | **riverlands**<br>continents laced with rivers + lakes<br>![rivers](superearth/se_rivers.png) | **arid**<br>desert continents (vegetation off)<br>![arid](superearth/se_arid.png) |
| **living** (night)<br>bioluminescence + city lights<br>![living](superearth/se_living.png) | **flat** (mountains off)<br>the with/without-mountains knob<br>![flat](superearth/se_flat.png) | **moons**<br>configurable system (rocky/icy/lava/desert)<br>![moons](superearth/se_moons.png) |

### Super-planets — higher degrees of freedom

| | | |
|---|---|---|
| **gas giant**<br>banded zonal flow + great red spot<br>![gas](superearth/se_gas.png) | **windstorm**<br>turbulent bands + cyclone eyes<br>![windstorm](superearth/se_windstorm.png) | **electrostorm**<br>slate thunderheads + lightning<br>![electrostorm](superearth/se_electrostorm.png) |

Lightning crackling across the electrostorm:

![electrostorm](superearth/se_electrostorm.gif)

### Orbital bombardment (`se_nuked`)

Configurable strikes — warhead **count**, spatial **distribution** (uniform /
clustered / equatorial / spiral), **delay**, **interval**, and detonations **in
parallel** per wave. Fireballs cool through a blackbody ramp; each strike leaves
an expanding shock-ring scar.

![se_nuked](superearth/se_nuked.png)

![se_nuked animated](superearth/se_nuked.gif)

## Cosmos & bodies

| | | |
|---|---|---|
| **earth** (from space)<br>![earth](earth.png) | **planet** (lit + lens flare)<br>![planet](planet.png) | **sun** (turbulent corona)<br>![sun](sun.png) |
| **black hole** (lensed disk)<br>![black hole](black-hole.png) | **quark**<br>![quark](quark.png) | **atom** (hydrogen)<br>![atom](atom.png) |
| **proton**<br>![proton](proton.png) | **neutron**<br>![neutron](neutron.png) | **electron** (1s cloud)<br>![electron](electron.png) |

Also in this family: `neutron_star` (pulsar with relativistic jets) and
`starfield` (a minimal registry demo).

## Elements (stylized Bohr atoms)

Twenty elements (H through Ar) live in `scenes/elements.py`, each a `Scene` in
the shared `SCENES` list — `python render.py --scene carbon`, `--scene argon`, …

| | | | |
|---|---|---|---|
| hydrogen<br>![H](elements/hydrogen.png) | helium<br>![He](elements/helium.png) | carbon<br>![C](elements/carbon.png) | oxygen<br>![O](elements/oxygen.png) |
| neon<br>![Ne](elements/neon.png) | sodium<br>![Na](elements/sodium.png) | chlorine<br>![Cl](elements/chlorine.png) | argon<br>![Ar](elements/argon.png) |

## Physics simulations

Real Warp particle physics (gravity, buoyancy, drag) driving nuclear /
thermonuclear blasts and Earth-impact scenarios — see `warp_shaders/sim/`.

| | | |
|---|---|---|
| **nuclear** blast chain<br>![nuclear](sim/nuclear.png) | **thermonuclear** blast<br>![thermonuclear](sim/thermonuclear.png) | **earth impact** (grounded)<br>![impact](sim/earth_grounded.png) |

Animated: [`nuclear.gif`](sim/nuclear.gif), [`thermonuclear.gif`](sim/thermonuclear.gif).

## The solar system — the namesake, configurable end to end

One renderer (`warp_shaders.cosmos`) draws any mix of stars (sun / neutron star /
white dwarf / black hole) and configurable planets on chosen Keplerian orbits,
plus an optional nebula. `--frames` animates the orbits. See
[Research 10](research/10-solar-system.md).

The celestial bodies (each a reusable shader):

| | | | |
|---|---|---|---|
| **sun**<br>granulation + spots + corona<br>![sun](cosmos/body_sun.png) | **neutron star**<br>compact + twin pulsar beams<br>![neutron](cosmos/body_neutron.png) | **white dwarf**<br>hot blue-white remnant<br>![dwarf](cosmos/body_dwarf.png) | **black hole**<br>lensing + disk + photon ring<br>![blackhole](cosmos/body_blackhole.png) |

The systems:

| | | |
|---|---|---|
| **solar_system** (the first)<br>neutron star + a planet on an ellipse<br>![solar_system](cosmos/solar_system.png) | **binary**<br>two suns + a planet around the pair<br>![binary](cosmos/ss_binary.png) | **trinary**<br>sun + neutron star + white dwarf + gas giant<br>![trinary](cosmos/ss_trinary.png) |
| **black-hole system**<br>a hole lensing a companion sun + planet<br>![blackhole](cosmos/ss_blackhole.png) | **nebula cradle**<br>a sun + two planets inside a nebula<br>![nebula](cosmos/ss_nebula.png) | **positioned nebula**<br>filamentary volume, placed + sized<br>![nebula](cosmos/body_nebula.png) |

The first system orbiting, and a **collapsing** system — two suns spiral in,
merge, collapse to a black hole (supernova), which swallows the planet:

![solar system orbiting](cosmos/solar_system.gif)
![collapsing system](cosmos/ss_collapse.gif)

### Cinematics — camera paths + the reel

`ss_flyby` sweeps a **keyframed Catmull-Rom camera** around the trinary system
(orbit + elevation ease + push-in, looping). Author moves with
[`engine.camera_path`](api/engine.md#shots--enginecamera_path), render sequences
to **MP4 / WebP / GIF** (`render.py --video`, [`engine.video`](api/engine.md#video--enginevideo)),
and stitch scenes into a showcase with [`reel.py`](guides/cinematics.md).

![cinematic fly-by of the trinary system](cosmos/ss_flyby.gif)

```bash
python render.py --scene ss_flyby --frames 144 --fps 24 --video out/flyby.mp4
python reel.py -o out/showcase.mp4 --width 960 --height 540 --fps 24
```

## The stellar life-cycle — one star, birth to remnant

One star evolving on a normalized timeline (`cosmos.stellar_evolution`): a
protostar condenses in a dusty **cradle**, settles onto the **main sequence**,
swells into a **red giant**, and ends — for a Sun-like star as a **planetary
nebula + white dwarf**, for a massive star as a **supernova + neutron star or
black hole**. Each frame carries a live **H-R diagram** inset tracing the star's
path across temperature/luminosity. See [Research 11](research/11-stellar-evolution.md).

![a Sun-like star's whole life, with the H-R diagram inset](cosmos/stellar_lifecycle.gif)

The **mass fork** — same code, three initial masses, three endings:

| | | |
|---|---|---|
| **stellar_lifecycle** (1 M☉)<br>→ planetary nebula → white dwarf | **stellar_massive** (14 M☉)<br>→ supernova → neutron star | **stellar_collapse** (30 M☉)<br>→ supernova → black hole |

```bash
python render.py --scene stellar_lifecycle --frames 120 --fps 6 --video out/life.mp4
python render.py --scene stellar_massive --time 14 -o supergiant.png
```

## Colliding galaxies — bridges and tails

The largest set-piece: two galaxies in a gravitational fly-by, modelled as a
**Toomre restricted N-body** encounter (`cosmos.galaxy_dynamics`) — two point-mass
cores under mutual gravity, each ringed by thousands of **massless test
particles**. A **prograde** disk throws out a long tidal **tail** and a **bridge**
toward the companion (the Antennae / Mice look); a **retrograde** disk barely
responds — the classic Toomre & Toomre 1972 contrast. Each galaxy's stars keep
their colour, so the tidal debris stays legible. See
[Research 12](research/12-galaxy-collisions.md).

![two galaxies colliding, tidal tails unfurling](cosmos/galaxy_collision.gif)

```bash
python render.py --scene galaxy_collision  --frames 64 --fps 6 --video out/tails.mp4
python render.py --scene galaxy_retrograde --frames 64 --fps 6 --video out/retro.mp4
```
