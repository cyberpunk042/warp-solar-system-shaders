# Gallery

Every scene is one module in `warp_shaders/scenes/`, rendered with
`python render.py --scene NAME --quality high -o out.png`. Run
`python render.py --list` for the full, current list (66 scenes).

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
| **underwater reef**<br>rippling caustics, blue-green depth, god-rays<br>![reef](engine/reef.png) | | |

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
