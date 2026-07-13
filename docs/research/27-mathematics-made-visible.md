# Research 27 — Mathematics made visible

> Pure mathematics has a shape. Chaos draws butterfly wings, topology folds a
> surface through itself, four dimensions cast a shadow into three, and a simple
> rule tiles the plane forever without repeating. Making the invisible visible.

## Strange attractors — deterministic chaos

A **strange attractor** is the set a chaotic dynamical system settles onto: not a
point, not a loop, but a fractal curve of infinite length in finite space. Tiny
changes in start conditions diverge exponentially (the *butterfly effect*), yet the
trajectory is confined forever to the attractor's shape. Integrate the ODE and draw
the path:

| Attractor | Equations (ẋ, ẏ, ż) | Look |
|---|---|---|
| **Lorenz** (1963) | σ(y−x), x(ρ−z)−y, xy−βz; σ=10, ρ=28, β=8/3 | the butterfly / owl mask |
| **Aizawa** | (z−b)x − dy, dx + (z−b)y, c + az − z³/3 − (x²+y²)(1+ez) + fzx³ | a ribbed spindle/shell |
| **Thomas** | sin y − bx, sin z − by, sin x − bz; b=0.208 | a symmetric looping lattice |

We integrate a swarm of points with a small time-step (RK/Euler) and additively
draw their trail — a glowing wire of chaos.

## Topology — surfaces that fold through themselves

- **Torus knot** — a curve winding *p* times around a torus's axis and *q* times
  through its hole (a (p,q) knot); (2,3) is the trefoil. A closed knotted tube.
- **Klein bottle** — a closed surface with no inside or outside: its neck passes
  *through* its own wall (only possible without self-intersection in 4D). The
  figure-8 immersion is the classic 3D shadow.

## Four dimensions — the tesseract

A **tesseract** (4-cube) has 16 vertices in 4-space. Rotate it in a 4D plane and
**project** to 3D (then to 2D): the inner cube appears to turn inside-out through the
outer one — a shadow of a rotation we cannot see directly, the way a wireframe cube
on paper is a shadow of a 3-cube.

## Aperiodic order — Penrose tiling

A **Penrose tiling** covers the plane with two rhombi (thin + thick) so that the
pattern **never repeats** yet has perfect five-fold symmetry — *aperiodic order*.
Built by **deflation** (subdividing each tile into smaller ones) or as a projection
of a 5D lattice. Kites and darts; golden-ratio proportions everywhere.

## Cellular automata & complex maps

- **Rule 30 / elementary CA** — a 1D row of cells updated by a 3-neighbour rule;
  Rule 30 produces provable chaos from total simplicity (Wolfram). Drawn as a
  triangle of generations stacked downward.
- **Domain colouring** — visualise a complex function f(z) by colouring each point z
  of the plane: **hue** = arg f(z), **brightness** = |f(z)|. Zeros and poles show as
  colour-wheel pinwheels; e.g. z³−1 has three, (z⁴−1)/(z²+1) shows zeros *and* poles.

## Rendering approach

| Scene | Technique |
|---|---|
| **strange_attractor** | integrate Lorenz/Aizawa point-swarm on host, additively splat the glowing trajectory |
| **torus_knot** | SDF of a (p,q) knot tube, raymarched with PBR-ish emissive shading |
| **klein_bottle** | the figure-8 immersion as a parametric tube SDF / point cloud, orbiting |
| **tesseract** | 4D vertices + edges, 4D rotation matrix → project 4D→3D→2D, draw glowing edges |
| **penrose_tiling** | deflation of the P3 rhombi on host → screen-space fill + edge glow |
| **domain_coloring** | evaluate a complex rational f(z) per pixel → HSV(arg, |f|) with contour bands |

Reuses `subatomic.field.sd_capsule`, `engine.intersect`, `engine.post`,
`subatomic.render.orbit_camera`, and the screen-space kernel pattern.

## Citations

- E. Lorenz, *Deterministic Nonperiodic Flow*, J. Atmos. Sci. (1963).
- M. Aizawa; R. Thomas (1999) — cyclically-symmetric attractors.
- F. Klein (1882) — the Klein bottle.
- R. Penrose (1974); M. Gardner, *Scientific American* (1977) — Penrose tilings.
- S. Wolfram, *A New Kind of Science* (2002) — elementary CA / Rule 30.
- H. S. M. Coxeter, *Regular Polytopes* (1948) — the tesseract / 4D projection.
