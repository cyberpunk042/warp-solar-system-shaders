# Research 29 — Megastructures & the far future

> If a civilisation keeps growing, it eventually builds on the scale of planets, stars,
> and beyond. The engineering of Kardashev II–III: capturing a star's whole output,
> living inside spun-up worlds, and turning matter into thought.

## The Kardashev scale

Nikolai Kardashev (1964) ranked civilisations by energy use: **Type I** commands a
planet's energy, **Type II** a star's entire output (~10²⁶ W), **Type III** a galaxy's.
The megastructures below are the architecture of Types II and III.

## Dyson sphere / swarm

Freeman Dyson (1960): a Type II civilisation could surround its star with a **swarm**
of collectors (a rigid shell is unstable and materially impossible) to capture most of
its light. From outside, the star dims in visible light and the structure re-radiates
the absorbed energy as **waste heat in the infrared** — a Dyson sphere's telltale
signature. Rendered: a star wrapped in a shell/swarm of panels, glowing warm from
re-radiation.

## Ringworld & Niven ring

Larry Niven's **Ringworld** (1970): a band ~1 AU in radius spun around a star, its
inner surface a habitable strip millions of km wide with walls to hold the air, day and
night made by an inner ring of **shadow squares**. Spin provides gravity; you look
"up" and see the far side of the ring arching overhead across the sky.

## Rotating habitats

- **O'Neill cylinder** (Gerard O'Neill, 1976): a pair of counter-rotating cylinders
  ~8 km across and ~30 km long; **spin** presses you to the inner wall at 1 g. The land
  wraps overhead — a valley curving up into the sky — with alternating land strips and
  windows (for sunlight) running the length.
- **Generation ship**: a rotating-habitat starship carrying a whole ecosystem on a
  centuries-long voyage between stars — a world in a bottle, its inhabitants never
  seeing departure or arrival.

## Space elevator

A cable from the equator to beyond **geostationary** orbit (~36 000 km), held taut by a
counterweight so centrifugal force balances gravity — climbers ride it to orbit without
rockets. Requires a material (carbon nanotube / diamondoid) with enormous tensile
strength.

## Matrioshka brain

The ultimate Type II computer: nested Dyson shells, each running on the **waste heat**
of the shell inside it (a heat engine between the hot inner and cold outer shells),
extracting nearly all a star's energy as computation. A star turned into a mind —
"matrioshka" for the nested-doll shells.

## Rendering approach

| Scene | Technique |
|---|---|
| **dyson_sphere** | a star + a shell/swarm of collector panels (domain-repeated cells on a sphere, partial coverage) re-radiating warm IR |
| **ringworld** | a star with a thin bright ring at ~AU scale seen at a shallow angle, the far arc rising overhead, shadow squares |
| **oneill_cylinder** | interior view: land/window strips wrapping up and over the head, a spindle sun, the valley curving into the sky |
| **space_elevator** | a planet limb with a cable rising from the equator to a counterweight, a climber, stars |
| **generation_ship** | a rotating habitat starship in interstellar space (cylinder + rings + engine glow) against the star field |
| **matrioshka_brain** | concentric glowing shells around a star, cooler outward (hot white → deep IR red), a computing megastructure |

Reuses `cosmos` star shaders, `engine.color.blackbody`, `engine.intersect`,
`earthgfx.stars`, `subatomic.field.sd_capsule`, and `engine.post`.

## Citations

- N. Kardashev, *Transmission of Information by Extraterrestrial Civilizations* (1964).
- F. Dyson, *Search for Artificial Stellar Sources of Infrared Radiation*, Science (1960).
- L. Niven, *Ringworld* (1970).
- G. K. O'Neill, *The High Frontier* (1976).
- R. Bradbury (Robert), *Matrioshka Brains* (1997–99).
- Isaacs et al. (1966); Artsutanov (1960) — the space elevator / skyhook.
