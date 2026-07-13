# Research 42 — Gravitational lensing (ray-tracing a black hole for real)

> The engine already had a `black_hole` scene that *approximated* lensing with an analytic
> deflection. This is the real thing: every camera ray is treated as a **photon** and its path is
> **integrated through curved spacetime**. Light near a black hole doesn't travel in straight
> lines — so we don't pretend it does. The result is the image *Interstellar* made iconic, and it
> falls out of the physics rather than being painted on.

## The geodesic a photon actually follows

Around a non-spinning (Schwarzschild) black hole of mass `M`, light follows a **null geodesic**.
Working in geometric units where the Schwarzschild radius `r_s = 2GM/c² = 1`, a compact and
faithful way to integrate it in Cartesian coordinates is

```
d²x/dλ²  =  −(3/2) · h² · x / r⁵ ,   h = |x × v|   (conserved),   r = |x|
```

The `1/r⁵` "force" is **not** Newtonian gravity (that would be `1/r²`); the extra factor is
exactly what bends light by twice the Newtonian amount — the `3M/r` term of general relativity
that Eddington measured at the 1919 eclipse. `h` (the photon's angular momentum, i.e. its impact
parameter) is conserved, so it is computed once from the initial ray and held fixed while the
position and velocity are marched forward with an adaptive step (finer near the hole).

Three things can happen to a ray:

- **It falls in.** If `r` drops below the horizon (`r_s`), the photon is captured — this pixel is
  part of the **black shadow**.
- **It escapes.** If `r` grows past a large radius, the photon flew off to infinity; we read the
  **background starfield** in its *final* (bent) direction — which is why the stars behind the
  hole are smeared into arcs. That is gravitational lensing.
- **It grazes and loops.** Rays passing near the **photon sphere** (`1.5 r_s`) wind part-way
  around before escaping or falling in, piling up into the razor-thin bright **photon ring** at
  the shadow's edge.

The signature *Interstellar* look — the disk arcing **over the top and under the bottom** of the
shadow — is not two disks. It is the **far side** of the single flat disk, its light bent up and
over the hole and back down to the camera. A straight-line renderer physically cannot produce it;
a geodesic tracer produces it automatically.

## The accretion disk, physically

A thin disk in the equatorial plane, emitting where a ray crosses it between the ISCO and the
outer edge:

- **Temperature** follows the Shakura–Sunyaev thin-disk law `T ∝ r^-3/4` (hotter inward),
  coloured by the **blackbody** locus (`engine.color.kelvin_to_rgb`), warm-biased for the
  butterscotch Gargantua palette.
- **Relativistic Doppler beaming**: the disk orbits at a large fraction of `c`; the side rotating
  **toward** the camera is brightened and blueshifted (intensity `∝ D³`), the receding side dimmed
  and reddened. The Keplerian speed `β = √(GM/r)` rises toward the inner edge.
- **Gravitational redshift**: light climbing out of the potential loses energy by `√(1 − r_s/r)`,
  dimming and reddening the innermost disk.
- The disk is treated as semi-transparent, so the near edge lets the lensed far arc show through
  behind it.

## Scene

`gargantua` — a Schwarzschild black hole traced by real geodesic integration: lensed accretion
disk over the shadow, photon ring, Doppler-beamed + redshifted blackbody disk, lensed starfield.
Because it is deterministic (one integrated path per pixel, no Monte-Carlo), it renders fast and
animates cleanly — the camera can orbit and the lensing shifts correctly.

## Sources

- **Schwarzschild geodesics / the `1/r⁵` photon equation** — Riazuelo, *Seeing relativity*;
  rantonels' *Schwarzschild geodesics for the layperson* (the Cartesian formulation used here).
- **The Interstellar image** — James, von Tunzelmann, Franklin & Thorne, *Gravitational lensing by
  spinning black holes in astrophysics, and in the movie Interstellar* (Class. Quantum Grav. 2015).
- **Thin accretion disk** — Shakura & Sunyaev (1973); relativistic beaming/redshift — Luminet,
  *Image of a spherical black hole with thin accretion disk* (1979), the first such computation.
- **Blackbody colour** — reused from `engine.color` (Tanner-Helland Planckian-locus fit).
