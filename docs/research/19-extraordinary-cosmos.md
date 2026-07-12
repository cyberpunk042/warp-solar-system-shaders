# 19 · Extraordinary cosmos — wormholes, quasars, a devoured star

The [cosmos](../api/cosmos.md) engine already renders stars, a gravitationally
lensed [black hole](15-nuclear-fireball.md), nebulae, galaxies, binaries, mergers
and a configurable solar system. This arc adds three of the most **extraordinary**
objects and events in the universe — each built on the same reusable GR photon
integrator (`cosmos.blackhole.bh_pixel` / `bh_escape_dir`) that already bends light
around the black hole.

## Wormhole (Ellis / Morris–Thorne)

A traversable wormhole is a throat connecting two regions of spacetime. The
simplest is the **Ellis drainhole** (Ellis 1973), with the metric

    ds² = −dt² + dl² + (b² + l²)(dθ² + sin²θ dφ²)

where `l ∈ (−∞, ∞)` is the proper radial coordinate and `b` is the **throat
radius**: at `l = 0` the two-sphere reaches its minimum area `4πb²` — the throat.
Morris & Thorne (1988) showed such a geometry is traversable only if threaded by
**exotic matter** (violating the null energy condition) to hold the throat open;
this is the physics *Interstellar* (Kip Thorne's *The Science of Interstellar*,
2014) popularised.

Visually the signature is unmistakable: a **sphere that shows another universe**.
Light rays that miss the throat are gravitationally lensed and wrap this universe's
sky into an **Einstein ring** around the rim; rays that enter the throat emerge in
the *other* universe and paint its sky, fish-eye-distorted, across the disc. We
render exactly that: near-miss rays are bent (reusing the deflection integrator)
and sample **background A**; rays that cross the throat radius sample a different
**background B** (a second nebula + starfield), with an exotic-matter rim glow.

## Quasar — relativistic jets from an active galactic nucleus

A **quasar** is a supermassive black hole (10⁶–10¹⁰ M☉) accreting so fiercely that
its **accretion disk** outshines the host galaxy. Perpendicular to the disk, twin
**relativistic jets** are launched — plasma collimated by the magnetic field to
within a few degrees and accelerated to Lorentz factors Γ ~ 10, radiating
**synchrotron** light (blue-white, polarised) punctuated by **shock knots** where
the flow re-brightens (the canonical example: the 5,000-light-year jet of M87,
imaged by the EHT alongside its black-hole shadow, 2019). The jet pointing toward
us is **Doppler-beamed** far brighter than its receding twin.

We extend the black-hole render: on top of the Doppler-beamed disk, two collimated
cones along the spin axis accumulate synchrotron emission with periodic shock knots
drifting outward, the approaching jet beamed brighter — an AGN core.

## Tidal disruption event — a star spaghettified

When a star wanders within the **tidal radius** `r_t ≈ R⋆ (M_BH / M⋆)^{1/3}` of a
black hole, the hole's tidal field exceeds the star's self-gravity and pulls it
apart — Hills (1975); Rees (1988). The star is stretched into a thin **stream** of
debris ("spaghettification"), half of which falls back and circularises into a
transient accretion disk, lighting a months-long **flare** that can outshine the
whole galaxy. We render the star elongating into a curved stream that winds into
the hole, brightening into a flare as it is devoured — an animated event.

## Reuse (not reinvention)

| Piece | From |
|---|---|
| GR photon-orbit deflection (ray bending) | `cosmos.blackhole.bh_pixel` / `bh_escape_dir` |
| Doppler-beamed accretion disk | `cosmos.blackhole._disk_emission` |
| blackbody colour | `engine.color.kelvin_to_rgb` |
| starfield + procedural nebula backgrounds | `earthgfx.stars`, `procedural.noise.fbm3` |

Nothing here is new gravitational physics — it is the same light-bending and
blackbody machinery, aimed at three new subjects.

## References

- H. G. Ellis, *Ether flow through a drainhole*, J. Math. Phys. 14 (1973).
- M. Morris & K. Thorne, *Wormholes in spacetime…*, Am. J. Phys. 56 (1988).
- K. Thorne, *The Science of Interstellar* (2014) — the film's wormhole/black-hole
  visualisations.
- M. Rees, *Tidal disruption of stars by black holes*, Nature 333 (1988);
  J. Hills, Nature 254 (1975).
- Event Horizon Telescope, *First M87 results* (2019) — the shadow + the jet.
- Research [15 · Nuclear fireball](15-nuclear-fireball.md) — the blackbody model reused here.
