# Gallery

Every scene is one module in `warp_shaders/scenes/`, rendered with
`python render.py --scene NAME --quality high -o out.png`. Run
`python render.py --list` for the full, current list (56 scenes).

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

## Life — grown from L-Systems

Real plants grown from L-System grammars (`warp_shaders.life`), tessellated to
a mesh and **ray-cast** through the Warp engine. Generation advances with
`time`, so they grow. See [Research 04](research/04-lsystems.md).

| | | |
|---|---|---|
| **grass**<br>tuft of arching blades<br>![grass](life/grass.png) | **herb**<br>stochastic leafy plant, golden-angle leaves<br>![herb](life/herb.png) | **tree**<br>parametric tapering tree + leafy canopy<br>![tree](life/tree.png) |
| **fern**<br>bracketed frond unfurling into a fiddlehead (ABOP fig 1.24)<br>![fern](life/fern.png) | | |

Growth (`--frames 8 --fps 1`), sprout → tree, and the fern unfurling:

![tree growing](life/tree_grow.gif)
![fern unfurling](life/fern_grow.gif)

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
