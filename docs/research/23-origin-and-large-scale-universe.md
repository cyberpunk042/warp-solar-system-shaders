# Research 23 — The origin & large-scale structure of the universe

> The other bookend to the sub-atomic strand: the very beginning and the very
> largest scales. What the Big Bang, the CMB, and the cosmic web *are*, and how we
> render them.

## The timeline

| Epoch | Time after Big Bang | What happens |
|---|---|---|
| **Inflation** | 10⁻³⁶–10⁻³² s | space expands exponentially (×10²⁶); quantum fluctuations stretched to cosmic scale — the seeds of all structure |
| **Quark–gluon plasma** | < 10 µs | too hot for protons; a soup of free quarks and gluons |
| **Nucleosynthesis** | ~3 min | protons + neutrons fuse → H, He, a little Li |
| **Recombination** | ~380,000 yr | plasma cools to ~3000 K; electrons bind to nuclei; the universe becomes **transparent** — releasing the **CMB** |
| **Dark ages** | 0.38–~100 Myr | neutral gas, no stars yet |
| **First stars** (Pop III) | ~100–400 Myr | gravity collapses the densest gas into the first, massive, metal-free stars — cosmic dawn |
| **Structure formation** | 0.4 Myr → now | dark-matter halos merge along filaments into the **cosmic web** of galaxies, clusters and voids |

## The cosmic microwave background (CMB)

The afterglow of recombination, redshifted from 3000 K to **2.725 K** today — a
near-perfect blackbody filling the whole sky, with **tiny temperature anisotropies**
(ΔT/T ~ 10⁻⁵): the density fluctuations that grew into galaxies. The Planck map is
the iconic mottled-oval all-sky image. We render it as a **fluctuation field on a
sphere** (fBm), false-coloured cold-blue → hot-red like the real maps.

## The cosmic web

On the largest scales matter is not uniform: dark-matter **filaments** thread
between dense **nodes** (galaxy clusters), bounding enormous near-empty **voids** —
a foam-like web (the Millennium / Illustris simulations). We render it as a 3-D
**Worley (cellular) noise** field — the cell *edges* are the filaments, the cell
*centres* the voids — with bright emissive knots at the vertices.

## First stars & structure growth

- **Population III stars** — the first stars, formed from pristine H/He with no
  metals, so they were very massive and blue-hot, igniting the dark universe.
- **Structure formation** — gravity amplifies the initial fluctuations: a nearly
  smooth field sharpens over time into the filamentary web. We animate this by
  ramping the contrast / sharpening of the density field with `time`.

## Rendering approach

| Scene | Technique |
|---|---|
| **big_bang** | a full-frame expanding hot **plasma** whose blackbody temperature cools (white→blue→red→dark) as the scale factor grows, seeded with fBm fluctuations + an initial flash |
| **cmb** | an fBm temperature field mapped onto a sphere, false-coloured (Planck palette) with a faint dipole |
| **cosmic_web** | ray-marched **Worley-edge** emission — filaments + bright cluster knots over voids, on a deep starfield |
| **first_stars** | blue-hot Population-III stars igniting one by one in a dark collapsing gas cloud |
| **structure_formation** | the density field sharpening from smooth to filamentary over `time` |

Reuses `engine.color` (blackbody), `procedural.noise` (fBm + Worley), `engine.post`,
and the `earthgfx.stars` void.

## Citations

- Planck Collaboration, *Planck 2018 results VI. Cosmological parameters*, A&A 641
  (2020) — CMB, ΔT/T ~ 10⁻⁵, T₀ = 2.725 K.
- A. Guth, *Inflationary universe*, Phys. Rev. D23 (1981) — cosmic inflation.
- V. Springel et al., *Simulations of the formation of the cosmic web* (Millennium),
  Nature 435 (2005) — filaments, nodes, voids.
- R. Barkana & A. Loeb, *In the beginning: the first sources of light*, Phys. Rept.
  349 (2001) — Population III stars, cosmic dawn.
