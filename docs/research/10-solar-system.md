# Research 10 — The solar system: stars, orbits, gravity, and light-bending

Sources and reasoning behind `warp_shaders/cosmos/` — the project's namesake: a
configurable solar system of 1–7 stars (sun / neutron star / white dwarf / black
hole) and configurable planets on chosen orbits, with a stable Kepler regime and
a destructive N-body one, plus an optional nebula.

## Celestial bodies as emissive spheres

Stars **emit**; they are not PBR-lit like planets, so each is a sphere shaded
directly by `bodies.shade_body` and composited by depth. One `StarConfig`
(kind / radius / temperature / activity / spin / precession) drives them all.

- **Sun** — the visible surface is the **photosphere**, a boiling layer of
  **convection cells** ("granulation"): hot plasma rises in bright cell centres
  and sinks in the darker inter-granular lanes. We render that with Worley
  cellular noise — specifically `F2 − F1`, which is large in a cell's interior
  and ~0 on the boundaries, so it lights the *cells* with dark lanes between
  them (the same trick the super-earth uses for rivers). Plus **sunspots** (cool
  magnetic dark patches), **faculae**, and **limb darkening** — the disk edge is
  dimmer and redder because a grazing sight-line sees higher, cooler plasma
  (a standard stellar-atmosphere effect). Colour comes from a blackbody-style
  temperature ramp (red dwarf → yellow sun → blue giant); the values are kept
  red-dominant so the warm hue survives the ACES tonemap instead of clipping to
  white.
- **Neutron star** — a city-sized remnant, blistering hot and blue-white, whose
  signature is the **pulsar lighthouse**: charged particles are flung along the
  magnetic axis into two narrow **polar beams**, and because the magnetic axis is
  tilted from the spin axis it **precesses** (wobbles). We render the beams
  analytically as the exact closest distance between each view ray and the beam
  axis line, so they read as sharp cones sweeping through space.
- **White dwarf** — an earth-sized, degenerate-matter remnant: small, very hot,
  smooth, blue-white, with a tight glow.
- **Black hole** — see the lensing section below.

## Orbits — Kepler for the stable regime

A planet on a fixed orbit is described by six **Keplerian elements**: semi-major
axis *a*, eccentricity *e*, inclination, longitude of the ascending node,
argument of periapsis, and a phase. To place it at time *t* we advance the
**mean anomaly** *M* linearly, then solve **Kepler's equation** *M = E − e·sin E*
for the **eccentric anomaly** *E* (Newton's method — it has no closed form), and
read off the position on the ellipse. This is exact two-body motion (Kepler's
1st–3rd laws, 1609–1619; the equation is Kepler 1621): the orbit closes every
period with no integration drift, so a laid-out system orbits forever.

## Gravity — N-body for the destructive regime

When bodies should actually *interact*, we integrate them as point masses under
mutual Newtonian gravity (**F = G·m₁m₂/r²**) with a softening length, using
**velocity-Verlet** (a symplectic "kick-drift-kick" integrator that conserves
energy far better than plain Euler over many steps). Bodies seeded from their
orbits and then slowed slightly **inspiral** and collide.

### Mergers and the correct remnant

When two stars touch they **merge** (conserving momentum). What the merged mass
*becomes* follows real stellar-remnant physics, dramatised:

- below the **Chandrasekhar limit** (~1.4 M☉) a degenerate remnant is a **white
  dwarf** — electron degeneracy pressure holds it up (Chandrasekhar, 1931);
- above it, up to the **Tolman–Oppenheimer–Volkoff limit** (~2–3 M☉), neutron
  degeneracy pressure gives a **neutron star**;
- above the TOV limit nothing can hold it — it collapses to a **black hole**.

Our thresholds are scaled for a watchable sequence rather than literal solar
masses, but the *ordering* is the physics: as the merged mass climbs it steps
star → neutron star → black hole, with a **supernova flash** on each collapse.
A black hole then **swallows** anything crossing its horizon, adding the mass
and momentum to itself and growing (its Schwarzschild radius scales with mass).

## Black holes bend light

A black hole is rendered by integrating the **general-relativistic photon-orbit
equation** per pixel:

    d²x/dλ² = −3/2 · h² · x / |x|⁵,   h = x × (dx/dλ)  (conserved)

This is the null-geodesic equation in the Schwarzschild metric (the ½·(3/2)
factor is the GR light-bending term beyond Newtonian gravity). Because *h* (the
specific angular momentum) is conserved, integrating it cheaply reproduces the
real strong-lensing picture: rays that pass close wind around the hole, so we get
the **Einstein ring** and the **photon ring** for free, and rays crossing the
**event horizon** (r < Schwarzschild radius) are captured — the black disk. A hot
**accretion disk** in the equatorial plane is accumulated where the bent ray
crosses it, with a temperature gradient (inner blue-white → outer orange) and
**Doppler beaming** — the side orbiting toward us is brighter and bluer. Seeing
the far side of the disk lensed up-and-over the top, plus the bright asymmetric
photon ring, is exactly the appearance computed for the film *Interstellar*
(James, von Tunzelmann, Franklin & Thorne, *Class. Quantum Grav.* 2015). In a
system the hole lenses the **whole rendered scene** as a screen-space pass, so
its companion stars and planets warp into the ring around it.

## Compositing the system

Rendering is layered so it reuses the whole engine: one kernel draws the
**starfield + nebula + all stars** (with a depth buffer); each **planet** is
rendered by the super-earth `render_planet` and **billboarded** at its projected
screen position and size, lit toward the brightest star with the correct
day/night **phase** and depth-tested so transits and eclipses order right; then
the black-hole lensing pass; then bloom + ACES tonemap.

## Sources

- Johannes Kepler — the three laws of planetary motion (1609–1619) and Kepler's
  equation (1621); Newton, *Principia* (1687) for the inverse-square law.
- Velocity-Verlet / leapfrog symplectic integration (Verlet 1967) for N-body.
- S. Chandrasekhar, *The Maximum Mass of Ideal White Dwarfs* (1931) — the ~1.4 M☉
  limit; Tolman (1939) and Oppenheimer & Volkoff (1939) — the neutron-star mass
  limit.
- K. Schwarzschild (1916) — the metric; the null-geodesic / photon-orbit equation
  for gravitational lensing.
- O. James, E. von Tunzelmann, P. Franklin, K. S. Thorne, *Gravitational lensing
  by spinning black holes in astrophysics, and in the movie Interstellar*
  (Class. Quantum Grav. 32, 2015) — the accretion-disk + lensing appearance.
- The pulsar "lighthouse" model (Gold 1968) for the neutron star's beams.
- Stellar photosphere granulation (solar convection) and limb darkening; Worley,
  *A Cellular Texture Basis Function* (1996) for the cellular noise.
- Planck blackbody radiation for the star colour–temperature ramp.
