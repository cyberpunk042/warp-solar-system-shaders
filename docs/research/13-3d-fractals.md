# Research 13 — 3D fractals: the Mandelbulb and the Mandelbox

Sources and reasoning behind `warp_shaders/procedural/fractal.py` — two famous 3D
escape-time fractals rendered the way the engine already renders everything else:
**distance-estimated ray marching**. A fractal has no surface equation, but it
*does* admit a **distance estimator** (DE) — a function that lower-bounds the
distance from any point to the set — and a DE is exactly what a sphere-tracer
needs. So the same raymarch loop that draws the [terrain](../research/00-foundations.md)
and the [reflections](12-galaxy-collisions.md) scene draws infinitely detailed
fractal surfaces.

## The Mandelbulb (White & Nylander, 2009)

The 2D Mandelbrot set iterates `z → z² + c` over complex numbers. There is no
3D complex number, so Daniel White and Paul Nylander built a **triplex** algebra:
treat a 3D point in **spherical coordinates** and define its nth power by
*multiplying the radius to the power and multiplying the angles by n*
([Wikipedia](https://en.m.wikipedia.org/wiki/Mandelbulb),
[Hvidtfeldt "Syntopia" DE series V](http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/),
[Quilez](https://iquilezles.org/articles/mandelbulb/)):

```
r = |z|,  θ = acos(z_z / r),  φ = atan2(z_y, z_x)
z^n = r^n · ( sin(nθ)cos(nφ),  sin(nθ)sin(nφ),  cos(nθ) )
```

Iterating `z → z^n + p` (with `p` the sampled point as the constant `c`) and
tracking the running **derivative** `dr → n·rⁿ⁻¹·dr + 1` gives the analytic
distance estimate

```
DE = 0.5 · log(r) · r / dr
```

The classic **power 8** gives the bulbous, brain-coral look; other powers change
the lobe count and symmetry (power 2–12 morph the shape). The DE is a *lower
bound*, so the sphere-tracer steps by `DE` and never overshoots the surface.

## The Mandelbox (Tom Lowe, 2010)

The Mandelbox ([Hvidtfeldt DE series VI](http://blog.hvidtfeldts.net/index.php/2011/11/distance-estimated-3d-fractals-vi-the-mandelbox/),
[Wikipedia](https://en.wikipedia.org/wiki/Mandelbox)) iterates a **conditional
folding** map instead of a power:

```
z → scale · sphereFold( boxFold(z) ) + p
```

- **box fold**: for each component, reflect it back into `[-1, 1]`
  (`if z > 1: z = 2 − z; if z < −1: z = −2 − z`) — this makes the boxy, cubic
  structure.
- **sphere fold**: if `|z|` is inside a small radius, scale it *up* (invert the
  core); if inside a unit radius, scale it out — this carves the round hollows.
- **scale** (often `−1.5` … `3`) stretches space each step; the DE tracks the
  running derivative `dr → |scale|·dr + 1` and returns `DE = |z| / |dr|`.

The result is an endless nested architecture of boxes and spheres — the classic
"fractal spaceship" you can fly into forever.

## Orbit traps — colour from the iteration

A DE tells you *where* the surface is; **orbit traps** tell you what *colour* to
paint it ([Wikipedia](https://en.wikipedia.org/wiki/Orbit_trap),
[Hvidtfeldt DE II](http://blog.hvidtfeldts.net/index.php/2011/08/distance-estimated-3d-fractals-ii-lighting-and-coloring/)).
During the escape-time loop we record how close the orbit came to a chosen shape
— here the **minimum `|z|`** (distance to the origin) and the **iteration count**
at escape. Those two scalars, returned alongside the DE, drive the surface colour:
the min-radius trap gives the banded, iridescent shells, and the iteration count
tints the deep crevices — no texture needed, the structure *is* the colour.

## How it renders

`fractal.py` exposes `mandelbulb_de(p, power, iters)` and `mandelbox_de(p, scale,
iters)` as device `@wp.func`s returning the DE plus the orbit-trap scalars; the
scenes sphere-trace them, estimate the normal from the DE gradient, and shade with
the engine's soft shadows + AO + sky + post. The Mandelbulb scene **morphs its
power** over time (2 → 8) so the fractal grows lobes; the Mandelbox scene slowly
turns a fly-through. Both are single-ray-per-pixel — no global illumination — so
they cost the same as any other SDF scene.

## Cross-references

- [Research 00 — foundations](00-foundations.md): the sphere-tracing raymarch +
  SDF gradient normal this reuses.
- Distance estimation: [Hvidtfeldt "Syntopia" series](http://blog.hvidtfeldts.net/index.php/category/distance-estimation/),
  [Quilez Mandelbulb](https://iquilezles.org/articles/mandelbulb/).
- The fractals: [Mandelbulb](https://en.m.wikipedia.org/wiki/Mandelbulb) (White &
  Nylander 2009), [Mandelbox](https://en.wikipedia.org/wiki/Mandelbox) (Lowe 2010).
