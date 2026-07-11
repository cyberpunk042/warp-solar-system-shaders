# Research 11 — The stellar life-cycle: one star, birth to remnant

Sources and reasoning behind `warp_shaders/cosmos/stellar_evolution.py` — a single
star evolving through its whole life on a normalized timeline, driving the star
shaders from [Research 10](10-solar-system.md). Where the solar-system work showed
*many* bodies in space, this shows *one* body in **time**: a stellar nursery
condenses into a protostar, it settles onto the main sequence, swells into a
giant, and ends — for a low-mass star as a planetary nebula + white dwarf, for a
massive star as a supernova + neutron star or black hole.

## The Hertzsprung–Russell track

The **HR diagram** plots a star's surface **temperature** (hot on the left, by
convention) against its **luminosity**. A star's life is a *path* across it, and
that path is what we animate. The canonical low-mass track
([Swinburne COSMOS](https://astronomy.swin.edu.au/cosmos/h/hertzsprung-russell+diagram),
[Wikipedia: HR diagram](https://en.wikipedia.org/wiki/Hertzsprung%E2%80%93Russell_diagram),
[PSU Astro 801](https://courses.ems.psu.edu/astro801/book/export/html/1759)):

1. **Protostar → pre-main-sequence (Hayashi track).** A collapsing molecular-cloud
   fragment above ~0.075 M☉ becomes a protostar; **T Tauri** stars are the
   sun-like pre-main-sequence phase (< ~10 Myr old), cool and over-luminous,
   contracting down a nearly **vertical** line on the HR diagram (the Hayashi
   track) before hydrogen ignites
   ([Britannica: T Tauri](https://www.britannica.com/science/T-Tauri-star),
   [Wikipedia: T Tauri](https://en.wikipedia.org/wiki/T_Tauri_star)).
2. **Main sequence.** Core hydrogen fusion — the longest phase; the star sits at a
   fixed point set by its mass (the Sun: G-type, ~5772 K, 1 R☉, ~10 Gyr).
3. **Subgiant / Hertzsprung gap.** Core hydrogen exhausted, shell burning begins;
   the star crosses rightward at nearly constant luminosity — fast, so few stars
   are caught here (hence the "gap").
4. **Red-giant branch.** A hydrogen shell around an inert helium core swells the
   envelope to tens–hundreds of R☉; the surface cools to ~3500 K and reddens
   ([Wikipedia: Red giant](https://en.wikipedia.org/wiki/Red_giant)).
5. **Helium flash → horizontal branch.** Core helium ignites (a runaway *flash* in
   low-mass stars) and the star settles onto the **horizontal branch**, burning He
   in the core via the triple-α process — hotter and smaller again
   ([Wikipedia: Horizontal branch](https://en.wikipedia.org/wiki/Horizontal_branch)).
6. **Asymptotic giant branch (AGB).** Helium and hydrogen shells around an inert
   carbon-oxygen core swell the star a second time, now with **thermal pulses** and
   heavy mass loss.
7. **Planetary nebula + white dwarf.** The AGB star sheds its envelope into a
   glowing **planetary nebula**, exposing the hot degenerate core — a **white
   dwarf** (~Earth-sized, 10⁴–10⁵ K) that then cools and fades
   ([PSU: final stages](https://courses.ems.psu.edu/astro801/content/l6_p3.html)).

## The mass fork

The ending depends almost entirely on the star's **initial mass**
([UCF/Lumen: more massive stars](https://pressbooks.online.ucf.edu/astronomybc/chapter/22-5-the-evolution-of-more-massive-stars/),
[Max Planck: life cycle](https://www.mps.mpg.de/sage/life-cycle-stars)):

| Initial mass | Track | End state |
|---|---|---|
| ≲ 8 M☉ | red giant → AGB → planetary nebula | **white dwarf** |
| ~8–20 M☉ | red **supergiant** → core-collapse (Type II) supernova | **neutron star** |
| ≳ 20–25 M☉ | red supergiant → supernova, with fallback | **black hole** |

Massive stars evolve the same way "only faster and bigger": they fuse up to an
iron core, become **red supergiants** (Betelgeuse-scale, ~1000 R☉), then their
cores collapse and they explode as **Type II supernovae**, leaving a neutron star
or a black hole. The thresholds in the code (`WD_MAX = 8`, `NS_MAX = 20 M☉`) follow
the standard division; they are ZAMS-mass boundaries, distinct from the *merged*-mass
thresholds used by the destructive N-body driver in [Research 10](10-solar-system.md).

## Timescales — why we compress time logarithmically

Lifetimes scale steeply with mass: the Sun spends ~10 Gyr on the main sequence, a
5 M☉ star ~100 Myr, a 20 M☉ star only ~10 Myr — and the *late* phases (subgiant,
flash, planetary nebula, supernova) are geologically **instantaneous** by
comparison. Rendering real-time proportions would be one frame of drama in an hour
of a static dot. So the animation maps a **normalized timeline** `t ∈ [0, 1]` onto
the phases by **visual interest**, not by real duration — every distinct phase gets
screen-time, and the fast, dramatic ends (helium flash, ejection, supernova) are
deliberately given room. The HR-track inset is annotated with the phase name so the
compression is legible.

## How the render is driven

`phase_state(t, mass)` is a pure host function returning the star's parameters at
time `t`: the body **kind** (sun / neutron / white-dwarf / black-hole), **radius**,
**temperature** (0 = cool red … 1 = hot blue, feeding the blackbody ramp),
**activity**, an **envelope** descriptor (protostar cradle / none / planetary
nebula / supernova ejecta with an expanding radius), a **flash** intensity (the
supernova spike), and the current **HR coordinates** (temperature, log-luminosity)
+ phase name for the inset. The renderer then reuses the existing star library
(`bodies.shade_body`, `body_corona`, `pulsar_beams`) with that evolving
`StarConfig`, layers the envelope (a spherical emissive shell integrated like the
nebula in [Research 10](10-solar-system.md)), and composites the HR-diagram panel.
Nothing about the *shaders* is new — the novelty is putting one star on a clock.

## Cross-references

- [Research 10 — the solar system](10-solar-system.md): the star shaders, blackbody
  colour, remnant thresholds, and nebula integrator this builds on.
- HR diagram: [Swinburne COSMOS](https://astronomy.swin.edu.au/cosmos/h/hertzsprung-russell+diagram),
  [Wikipedia](https://en.wikipedia.org/wiki/Hertzsprung%E2%80%93Russell_diagram).
- Phases: [T Tauri](https://en.wikipedia.org/wiki/T_Tauri_star),
  [red giant](https://en.wikipedia.org/wiki/Red_giant),
  [horizontal branch](https://en.wikipedia.org/wiki/Horizontal_branch),
  [AGB](https://en.wikipedia.org/wiki/Asymptotic_giant_branch),
  [massive stars](https://pressbooks.online.ucf.edu/astronomybc/chapter/22-5-the-evolution-of-more-massive-stars/).
