# Research 31 — States of matter & phase transitions

> Solid, liquid, gas — and beyond: plasma, condensates, glasses. Matter reorganising
> itself as energy is added or removed, and the strange orderings in between.

## The familiar three (and the fourth)

Add energy and matter climbs the ladder: a **solid** (atoms locked in a lattice) melts
to a **liquid** (atoms mobile but touching), boils to a **gas** (atoms free and far
apart), and — ionised at high enough energy — becomes a **plasma** (electrons torn from
nuclei, a glowing conductive soup). Plasma is the most common state of ordinary matter
in the universe (every star). On Earth: lightning, flames, neon signs, a **Tesla-coil**
arc.

## Phase transitions & latent heat

At a transition (melting, boiling) added energy goes into breaking bonds, not raising
temperature — the **latent heat**. Boiling water stays at 100 °C while bubbles of vapour
nucleate, grow and rise. Freezing runs the film backward: a **crystallisation** front
sweeps through a melt, atoms snapping into the lattice and releasing heat (a snowflake,
a freezing pond, a supercooled solution flashing solid).

## Order: crystal vs glass

- A **crystal** is long-range **ordered** — atoms on a repeating lattice, sharp facets,
  a discrete diffraction pattern.
- A **glass** is an **amorphous** solid — frozen liquid disorder, no long-range order.
  Cool a liquid fast enough and it vitrifies instead of crystallising. Same chemistry
  (silica), utterly different order.

## Exotic states

- **Bose–Einstein condensate (BEC)** — cool bosonic atoms to nanokelvin and they
  collapse into a *single* quantum state, one giant matter-wave. Seen as a sharp central
  peak rising out of a thermal cloud (Cornell/Wieman/Ketterle, 1995).
- **Ferrofluid** — a colloid of magnetic nanoparticles that, in a magnetic field, grows
  a field of self-organising **spikes** (the Rosensweig instability) along the field
  lines — a liquid that spikes.

## Rendering approach

| Scene | Technique |
|---|---|
| **plasma_arc** | a Tesla-coil / Jacob's-ladder electric arc — branching glowing filaments (fractal lightning) between electrodes |
| **crystallization** | a crystallisation front sweeping a supercooled melt, dendritic ice fingers snapping outward |
| **ferrofluid** | a black magnetic fluid growing a hexagonal field of Rosensweig spikes along field lines |
| **boiling** | a rolling boil — vapour bubbles nucleating on a hot floor, growing, detaching and rising |
| **bose_einstein** | a thermal cloud cooling: a sharp condensate peak rising out of the broad Gaussian (the iconic BEC plot) |
| **glass_vs_crystal** | side-by-side atomic order — an ordered lattice vs an amorphous glass (same atoms, different order) |

Reuses `procedural.noise` (fbm, curl, worley), `procedural.hash`, `engine.color`,
`subatomic.field.sd_capsule`, and `engine.post`.

## Citations

- I. Langmuir (1928) — coined "plasma" for ionised gas.
- W. Ostwald; the Gibbs phase rule — phase equilibria & latent heat.
- M. Rosensweig, *Ferrohydrodynamics* (1985) — the ferrofluid spike instability.
- E. Cornell, C. Wieman, W. Ketterle (1995; Nobel 2001) — Bose–Einstein condensation.
- P. W. Anderson, *Through the Glass Lightly* (1995) — the nature of the glass transition.
