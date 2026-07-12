# Research 16 — a living world: life under its sun(s)

The three strands of this project have run in parallel: the **engine** (terrain,
atmosphere, sky), the **solar system** (`cosmos` — stars, orbits, seen from
space), and **aliveness** (`life` — L-System plants, ecosystems, the mind). This
arc **fuses** them. Instead of a planet as a distant billboard, we stand *on its
surface*: the living L-System ecosystem in the foreground, lit by the system's
actual **sun(s)** crossing the sky, with the day and the seasons driven by the
planet's rotation and orbit.

Design behind `life.render.render_world` and the `living_world` / `twin_suns`
scenes.

## What we already have to fuse

- **`life.render.render_plant`** ray-casts a merged L-System mesh with GGX PBR,
  a single sun, a cast shadow, a sky gradient + sun disc, and post — a full
  surface renderer. The `ecosystem` scene already varies the sun by **season**.
- **`cosmos`** knows real star **colours** (blackbody `kelvin_to_rgb`) and
  **orbits** (`cosmos.orbits`) — everything needed to place suns in a surface sky
  and colour their light.
- **`engine`** has atmospheric scattering, fog and the sky helpers.

The fusion is a renderer that takes **N suns** (direction + colour + intensity)
from a solar-system configuration, lights the ecosystem with all of them, casts a
shadow **per sun**, and paints a day/twilight sky with each sun's disc.

## The day — a sun arc

The sun's apparent motion is the planet's **rotation**. Over a day the sun climbs
from the dawn horizon to a noon maximum set by latitude + season, then sets. Two
things track it:

- **Sky colour** — at low sun elevation the path length through the atmosphere is
  long, so Rayleigh scattering reddens the light (dawn/dusk); near noon it is
  blue-white. We drive the sky tint and the sun's colour temperature from its
  elevation.
- **The life** — the ecosystem's plants are already **phototropic** (they bend
  toward the light) and fold/rest in low light, so a moving sun makes the meadow
  visibly lean and open through the day. The obvious rules, now under a real sun.

## The seasons — axial tilt, not distance

A common misconception is that seasons come from the planet being nearer/farther
from its star. They don't: they come from **axial tilt** (obliquity ≈ 23.4° for
Earth), which changes the **angle** at which sunlight strikes the ground — and the
day length — over the orbit
([Wikipedia — axial tilt](https://en.wikipedia.org/wiki/Axial_tilt),
[Milankovitch cycles](https://en.wikipedia.org/wiki/Milankovitch_cycles)). Higher
sun + longer days ⇒ more **insolation** ⇒ summer. We tie the ecosystem's existing
season phase to the planet's **orbital position**, and set the noon sun height
from the tilt, so the meadow's spring→winter recolouring is the *same* orbit the
`cosmos` scenes show from space.

## Two suns — a habitable "Tatooine"

**Kepler-16** is the first confirmed circumbinary system: a **K dwarf** (~0.69
M☉, orange) and a **red dwarf** (~0.20 M☉) in a 41-day orbit, with a planet
circling both — so from its surface you would see **two suns** and, at day's end,
**two sunsets**, exactly like Tatooine
([NASA/JPL](https://www.jpl.nasa.gov/news/nasas-kepler-discovery-confirms-first-planet-orbiting-two-stars/),
[Kepler-16b — Wikipedia](https://en.wikipedia.org/wiki/Kepler-16b)). The real
Kepler-16b is a cold gas giant, but a *habitable* version is the natural
life-bearing set-piece. Two suns of different colour and brightness produce:

- **Two shadows** per object, offset by the suns' angular separation, each tinted
  by the *other* sun's colour (a warm shadow under cool light, and vice-versa).
- **Combined illumination** — the diffuse light is the sum of both suns; where one
  sun is occluded (its shadow), the ground is still lit by the other, so shadows
  are soft and coloured, never black.
- **A double sunset** — as the pair nears the horizon the two discs redden and set
  minutes apart.

We render this as a binary in the surface sky: an orange K-dwarf sun + a cooler
white companion, each casting its own shadow across the living meadow.

## How it renders

`life.render.render_world(mesh, suns, ...)` takes a list of suns (each a
direction, colour, intensity). For every shaded point it sums `shade_pbr` over the
suns and traces one shadow ray per sun (so each sun contributes its own soft
shadow, and a point in one sun's shadow is still lit by the other). The sky is a
day-gradient (elevation-tinted) with each sun's disc composited in. The
`living_world` scene runs one sun across a day over the ecosystem meadow; the
`twin_suns` scene lights it with a Kepler-16-like pair for the double-shadow,
double-sunset look. Everything else — the L-System growth, phototropism, the
seasons — is the machinery already built; this arc just puts a real sky over it.

## Cross-references

- [Research 04 — L-Systems](04-lsystems.md), [08 — ecosystem](08-ecosystem.md):
  the life this world grows.
- [Research 10 — the solar system](10-solar-system.md),
  [11 — stellar evolution](11-stellar-evolution.md): the stars whose light now
  reaches the ground.
- Sources: [Kepler-16b](https://en.wikipedia.org/wiki/Kepler-16b),
  [NASA/JPL two-star discovery](https://www.jpl.nasa.gov/news/nasas-kepler-discovery-confirms-first-planet-orbiting-two-stars/),
  [axial tilt & seasons](https://en.wikipedia.org/wiki/Axial_tilt).
