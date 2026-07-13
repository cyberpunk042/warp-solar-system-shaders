# Research 40 — Physics simulations (gravity, fluids)

> Third of the four-part arc (electricity → engine leap → **physics sims** → the living body).
> The earlier strands *drew* physics; this one **runs** it. Instead of an analytic field baked
> into a shader, a state is stepped forward in time by the governing equations, on Warp, and the
> result is whatever the dynamics produce — emergent, not authored. Two classic simulations open
> the strand: **N-body gravity** and an **incompressible fluid**.

## N-body gravity

Newton's law of gravitation is pairwise: every body pulls on every other with a force along the
line joining them, proportional to the product of masses and inversely to the square of the
distance. For `N` bodies that is an **`O(N²)` force sum** each step — exactly the kind of dense,
uniform, parallel workload a GPU (and Warp on CPU) eats for breakfast: one thread per body sums
the pull of all the others.

- **Softening.** A raw `1/r²` diverges when two bodies pass close, blowing up the integrator.
  Replacing `r²` with `r² + ε²` (Plummer softening) caps the force at short range — the standard
  fix in collisionless stellar dynamics, where we don't want spurious two-body scattering anyway.
- **Leapfrog integration.** Positions and velocities are advanced in a *kick–drift* interleave.
  Leapfrog is **symplectic**: it conserves a nearby "shadow" energy exactly, so orbits don't
  spiral in or out from integration error the way naive Euler would — essential for a system
  meant to run for thousands of steps.
- **The physics that emerges.** Set two bound clumps on a grazing collision course and nobody
  scripts what happens next: they fall together, raise **tidal tails** and bridges from the
  differential pull across each cloud, and — through **violent relaxation** — phase-mix into a
  single smooth, centrally-concentrated remnant. This is, in miniature, how galaxies merge.

`nbody` renders exactly this: two Plummer spheres (cool-blue and warm-gold) colliding, particles
additively splatted so density reads as brightness and the merged core blazes.

## Incompressible fluids — Stable Fluids

Fluids obey the **Navier–Stokes equations**: momentum is advected by the flow itself (the
non-linear term that makes turbulence hard), pushed by pressure and buoyancy, and — for an
incompressible fluid — the velocity field must stay **divergence-free** (no fluid created or
destroyed). Jos Stam's *Stable Fluids* (1999) is the method that made real-time fluid animation
possible, by choosing an **unconditionally stable** discretisation of each term:

- **Semi-Lagrangian advection.** To move a quantity, don't push it forward (which can overshoot
  and explode); instead, for each cell trace the velocity *backward* a step and sample where the
  fluid came from (bilinear interpolation). Stable for any time-step.
- **Pressure projection.** After advection the field has divergence; solve a **Poisson equation**
  for a pressure whose gradient, subtracted off, removes exactly that divergence (a few dozen
  Jacobi iterations here). This is the incompressibility constraint made numerical.
- **Buoyancy + vorticity confinement.** Hot fluid is lifted (buoyancy); and because the coarse
  grid numerically *diffuses away* small-scale swirl, **vorticity confinement** measures the
  curl that remains and pushes energy back into it — restoring the curls and billows that read as
  "smoke" rather than a smooth smear.

`fluid` runs this on a grid with a hot emitter at the base: a turbulent, curling column rises,
grey smoke billowing off a glowing ember source.

## Cost and scaling

Both simulations are stepped from rest each frame here (stateless render), so a still is cheap
but a long animation is not — the N-body sum is `O(steps · N²)`, the fluid `O(steps · grid²)`.
On CPU that bounds the resolution; the exact same kernels scale straight up on CUDA, where the
`O(N²)` gravity sum and the grid solve are both ideal GPU workloads.

## Sources

- **N-body / softening / violent relaxation** — Aarseth, *Gravitational N-Body Simulations*;
  Binney & Tremaine, *Galactic Dynamics*.
- **Symplectic (leapfrog) integration** — Hairer, Lubich & Wanner, *Geometric Numerical
  Integration*.
- **Stable Fluids** — Jos Stam, *Stable Fluids* (SIGGRAPH 1999); *Real-Time Fluid Dynamics for
  Games* (GDC 2003).
- **Vorticity confinement** — Fedkiw, Stam & Jensen, *Visual Simulation of Smoke* (SIGGRAPH 2001).
