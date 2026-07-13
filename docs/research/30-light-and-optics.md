# Research 30 — Light & optics

> How light bends, splits, bounces and interferes. The physics behind rainbows,
> soap-bubble colours, laser fringes and the dancing light on a pool floor.

## Refraction & dispersion

Light slows in glass or water (refractive index *n* = c/v). At a surface it **bends**
by **Snell's law**: n₁ sinθ₁ = n₂ sinθ₂. Crucially *n* depends slightly on wavelength
(**dispersion**) — blue bends more than red — so a **prism** fans white light into a
spectrum. The same effect, in raindrops, makes the **rainbow**.

## The rainbow

A raindrop refracts sunlight entering, **totally internally reflects** it off the back,
and refracts it again leaving — bending it back by ~**42°** (the *rainbow angle*), with
each colour at a slightly different angle (red outer, violet inner). A fainter
**secondary** bow at ~51° comes from *two* internal reflections, with colours reversed,
and a dark **Alexander's band** sits between them.

## Thin-film interference

A thin film (soap bubble, oil on water) reflects light off both its front and back
surfaces. The two reflections **interfere**: where the path difference is a whole
wavelength they add (bright), half a wavelength they cancel (dark) — and since the
condition depends on wavelength and film thickness, you see swirling **iridescent**
colour bands that shift as the film thins.

## Diffraction

Light bends around edges and through slits, spreading and interfering. A **diffraction
grating** (thousands of fine lines) sends each wavelength to a different angle
(d sinθ = mλ) — splitting light into sharp spectral orders, the shimmer of a CD.

## Caustics

When a curved or rippled surface focuses light, the concentrated bright curves are
**caustics** — the net of shifting light on a pool floor, the bright cusp inside a
coffee cup. They are the envelopes of refracted/reflected rays, brightest where rays
pile up.

## Interference & the laser

Coherent light (a **laser** — stimulated emission, one wavelength, in phase) split and
recombined produces **interference fringes** (Young's double slit; the Michelson
interferometer). Bright and dark bands map path differences to a fraction of a
wavelength — the basis of holography and gravitational-wave detectors (LIGO).

## Rendering approach

| Scene | Technique |
|---|---|
| **prism** | a triangular glass prism refracting a white beam into a fanned spectrum (per-wavelength Snell bend) |
| **rainbow** | rain volume + the 42°/51° primary+secondary bows over a landscape, Alexander's band |
| **thin_film** | an iridescent soap-bubble / oil film (interference colour from thickness × angle) |
| **diffraction_grating** | a beam hitting a grating, split into spectral orders (d sinθ = mλ) |
| **caustics** | the shifting bright net of light refracted through a rippling water surface onto a floor |
| **interferometer** | a Michelson interferometer's concentric/striped interference fringes shifting |

Reuses `procedural.noise`, `engine.color` (wavelength→RGB), `engine.intersect`,
`engine.post`, and the screen-space + raymarch kernel patterns.

## Citations

- W. Snell (1621); R. Descartes, *Les Météores* (1637) — refraction & the rainbow angle.
- I. Newton, *Opticks* (1704) — dispersion, the spectrum, thin-film "Newton's rings".
- T. Young (1804) — the double-slit interference experiment.
- A. Michelson (1881) — the interferometer.
- M. Minnaert, *Light and Colour in the Open Air* (1954) — rainbows, caustics, halos.
