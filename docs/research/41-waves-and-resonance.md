# Research 41 — Waves, resonance & interference

> A companion to the physics-sims strand ([research 40](40-physics-sims.md)), all built on one
> equation. Where gravity and fluids each had their own dynamics, this strand is the many faces
> of a single one — the **wave equation** ``u_tt = c²∇²u`` — and its eigenmodes. Sand jumping to
> the still lines of a ringing plate, two ripples crossing, a drum singing its Bessel modes, a
> wave squeezing through two slits: all the same physics, seen four ways.

## Eigenmodes — why a bounded thing rings at special shapes

Clamp a wave medium at a boundary and it can no longer vibrate at just any frequency: only the
**normal modes** survive — standing-wave shapes whose boundary stays fixed. Each mode is an
eigenfunction of the Laplacian ``∇²`` with its own frequency. The mode's **nodal set** — the
curves that never move — is where loose material (sand, in cymatics) collects, because
everywhere else the surface is thrashing and shakes it away.

- **Square plate (Chladni).** The classic superposition
  ``cos(nπx)cos(mπy) − cos(mπx)cos(nπy)`` is (approximately) a free-plate mode; its zero
  contour is the intricate figure the sand draws. Raise the driving frequency and the plate
  jumps to a higher ``(n, m)``, the pattern re-forming — the effect Chladni demonstrated in 1787
  and Faraday, Rayleigh and Kirchhoff later explained.
- **Circular membrane (drum).** With a round clamped boundary the modes separate in polar
  coordinates: the radial part is a **Bessel function** ``J_m(k·r)`` (with ``k`` set so
  ``J_m(kR)=0`` — the skin is still at the rim) and the angular part is ``cos(mθ)``. Mode
  ``(m, n)`` has ``m`` nodal diameters and ``n−1`` nodal circles. This is literally why a drum's
  overtones are *not* harmonic — the Bessel zeros aren't evenly spaced.

## Interference — waves adding

Two waves at a point simply **add**. Where crest meets crest they reinforce (constructive); where
crest meets trough they cancel (destructive). For two coherent point sources the set of
cancellation points is a family of **hyperbolae** (constant path-difference), and the fixed dark
lines fan out between the sources — the ripple-tank pattern. Send a plane wave through **two
slits** and, by **Huygens' principle**, each slit becomes a new circular source; their
interference casts the bright/dark **fringes** of Young's experiment — the demonstration that
settled that light is a wave, and (with one quantum at a time) that matter is too.

## How this engine does it

Two engines, both trivially parallel and both matching the analytic physics:

- **Analytic eigenmodes** (`chladni`, `standing_membrane`). The mode shape is a closed-form
  function evaluated per pixel / per surface point — no time-stepping needed for a still. The
  drumhead is drawn as a true 3-D surface (painter's-ordered point splat) coloured by
  displacement; the plate is a top-down field with the sand banked on the nodal set.
- **Finite-difference wave equation** (`ripple_tank`, `double_slit`) — `sim/wave.py`. The field
  is stepped with the explicit leapfrog stencil ``u_next = 2u − u_prev + (c·dt/h)²∇²u``, point or
  line **oscillators** drive it, an optional **barrier mask** pins ``u = 0`` at wall cells, and a
  tapered border absorbs outgoing waves so the box doesn't ring. Stable while the **Courant**
  number ``c·dt/h ≤ 1``. Lit like a real ripple tank: brightness tracks the surface curvature
  (the Laplacian), which is what focuses light into the caustic fringes you actually see.

## Scenes

- `chladni` — cymatics: warm sand banked on the nodal lines of a vibrating square plate; the
  driving frequency sweeps up over frames so the resonance figure keeps dissolving and re-forming.
- `ripple_tank` — two in-phase sources radiating circular waves that interfere into fixed
  hyperbolic nodal lines, cyan caustics on dark water.
- `standing_membrane` — a circular drumhead frozen in a Bessel mode ``J_m(kr)cos(mθ)``, ray-drawn
  as a 3-D surface and coloured by displacement; the standing wave breathes in place.
- `double_slit` — a plane wave through two slits, the far-field fringe fan building as the wave
  arrives.

## Sources

- **Chladni figures / plate modes** — Chladni, *Entdeckungen über die Theorie des Klanges* (1787);
  Rayleigh, *The Theory of Sound*.
- **Membrane / Bessel modes** — Morse & Ingard, *Theoretical Acoustics*; Courant & Hilbert,
  *Methods of Mathematical Physics* (eigenvalues of the Laplacian).
- **Finite-difference wave equation, Courant condition** — Courant, Friedrichs & Lewy (1928);
  Strikwerda, *Finite Difference Schemes and PDEs*.
- **Interference / double slit** — Young (1804); Born & Wolf, *Principles of Optics*.
- **Bessel-function approximations** — Abramowitz & Stegun, *Handbook of Mathematical Functions*,
  §9.4 (the polynomial fits used here in lieu of SciPy).
