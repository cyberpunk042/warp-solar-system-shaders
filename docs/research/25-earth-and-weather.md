# Research 25 — Earth & weather systems

> The planet as a heat engine: the Sun drives the air and oceans, the interior
> drives the crust. What hurricanes, lightning, ocean currents, plate tectonics
> and the water cycle *are*, and how we render them.

## Hurricanes (tropical cyclones)

Over warm (≥26 °C) ocean, moist air rises, condenses (releasing latent heat) and
draws in more air; the Coriolis force spins the inflow into a huge rotating storm
with **spiral rainbands**, a warm calm **eye**, and a towering **eyewall** of the
fiercest wind. Rotation is counter-clockwise in the northern hemisphere. We render
the satellite top-down view: **log-spiral cloud bands** around a clear eye + bright
eyewall over ocean.

## Lightning

In a storm cloud, colliding ice and graupel separate charge — positive up, negative
down — building fields of ~10⁸ V until the air breaks down and a **stepped leader**
forks toward the ground; the return stroke flashes at ~30,000 K (hotter than the
Sun's surface) and the shock is thunder. We render dark **fBm storm clouds** lit
from within by branching bolts that flash on a timer.

## Ocean currents

The Sun heats the tropics unevenly; wind and density (temperature + salinity) drive
a planet-spanning flow — the **thermohaline circulation** ("great ocean conveyor"):
warm surface currents (Gulf Stream) poleward, cold dense water sinking and returning
at depth. We render **flowing streamlines** advected by a curl-noise field over the
globe, warm→cold coloured.

## Plate tectonics

Earth's crust is broken into **plates** riding the convecting mantle: they spread at
**mid-ocean ridges** (new crust, volcanism), collide at **subduction zones**
(trenches, arcs, quakes) and grind past at **transform faults**. We render a globe
with glowing **plate boundaries** (Worley-cell edges) and hot volcanic ridges.

## The water cycle

Solar heat **evaporates** ocean water; it rises, cools and **condenses** into clouds;
**precipitation** returns it as rain/snow; rivers carry it back to the sea. We render
the loop over a coastline: rising vapour → clouds → falling rain.

## Rendering approach

| Scene | Technique |
|---|---|
| **hurricane** | top-down log-spiral cloud bands + fBm detail around a clear eye + bright eyewall over ocean |
| **lightning_storm** | fBm storm clouds lit from within by branching bolts (recursive forks) flashing on a timer |
| **ocean_currents** | a globe with curl-noise-advected streamlines, warm→cold, over land/sea |
| **plate_tectonics** | a globe with glowing Worley-edge plate boundaries + hot volcanic ridges |
| **water_cycle** | a coastline with rising vapour, condensing clouds and falling rain, looping |

Reuses `procedural.noise` (fBm + Worley + curl), `engine.intersect`, `engine.post`,
and the Earth/atmosphere patterns.

## Citations

- K. Emanuel, *Divine Wind: the history and science of hurricanes*, OUP (2005).
- M. A. Uman, *The Lightning Discharge*, Academic Press (1987) — leaders, return
  strokes, ~30,000 K channel.
- W. Broecker, *The great ocean conveyor*, Oceanography 4 (1991) — thermohaline
  circulation.
- W. J. Morgan, *Rises, trenches, great faults, and crustal blocks*, JGR 73 (1968) —
  plate tectonics.
