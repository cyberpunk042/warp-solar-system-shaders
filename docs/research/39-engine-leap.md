# Research 39 — Engine leap (global illumination, path tracing)

> The engine has always been a **single-bounce** renderer: march a ray to the nearest
> surface, shade it with direct light + a few analytic tricks (AO, soft shadows, bloom),
> done. This strand adds the missing physics — **light that bounces**. A Monte-Carlo
> **path tracer** lets rays scatter around a scene many times, so colour bleeds between
> surfaces, shadows go soft and contact-tight for free, and everything is lit *consistently*
> by whatever emits. Second of the four-part arc (electricity → **engine leap** → physics
> sims → the living body).

## Why path tracing

Real light obeys the **rendering equation**: the radiance leaving a point is its own
emission plus the integral, over the hemisphere, of incoming radiance times the surface's
BRDF times the cosine term. That integral is recursive — incoming radiance is itself the
outgoing radiance of other points — so it has no closed form for a general scene. **Path
tracing** evaluates it by Monte-Carlo: shoot a ray, let it bounce randomly (importance-
sampling the BRDF), and every time a path reaches a light, deposit its throughput. Average
enough random paths per pixel and the noisy estimate converges to the true image.

What that buys, that a single-bounce renderer cannot fake:

- **Colour bleeding** — a red wall throws red light onto a neighbouring white surface.
- **Soft, contact-hardening shadows** — from an area light, for free, no shadow hack.
- **Consistent indirect light** — a room lit only by a small ceiling patch is fully,
  believably illuminated because light bounces into every corner.
- **Ambient occlusion for real** — corners darken because fewer bounce paths escape them,
  not because of a screen-space trick.

## How this engine does it

- **Sampling**: each pixel fires `spp` jittered primary rays; each path takes up to
  `_BOUNCES` diffuse bounces. Randomness comes from Warp's on-device RNG (`wp.rand_init` /
  `wp.randf`), seeded per pixel and per sample so every path is independent and the frame is
  deterministic.
- **Diffuse bounce**: at each hit the new direction is drawn **cosine-weighted** over the
  hemisphere about the surface normal (importance sampling the Lambertian BRDF, so the
  cosine and the pdf cancel and the estimator is just `throughput *= albedo`).
- **Geometry**: the scene is a signed-distance field (sphere-traced each bounce), so the
  same SDF toolkit that drives the rest of the engine drives the path tracer — no separate
  triangle pipeline. Normals come from the SDF gradient; the normal is flipped to face the
  incoming ray so interior walls light correctly.
- **Lights are geometry**: an emissive patch is just a surface whose emission is non-zero;
  a path that lands on it terminates and deposits `throughput × emission`. No separate light
  list, no next-event estimation yet (naive path tracing — simple, unbiased, a little noisy).

The cost is variance: the image is **noisy** until enough samples accumulate (noise falls as
1/√spp). On CPU that means dozens-to-hundreds of samples per pixel for a clean still; on GPU
the same kernel scales up directly.

## Scenes

`cornell_box` — the canonical global-illumination test: a five-wall open-front room (red left,
green right, white rest) with two white blocks, lit only by a ceiling emitter, path-traced so
the coloured walls bleed onto the blocks and floor with soft contact shadows. The "hello
world" of GI, and the proof the path tracer works.

*(more to come as the strand grows: a dielectric-glass / caustics showcase, subsurface
scattering, motion blur, and re-renders of flagship scenes at the new fidelity tier.)*

## Sources

- The **rendering equation** — Kajiya, *The Rendering Equation* (SIGGRAPH 1986).
- Monte-Carlo path tracing, importance sampling, cosine-weighted hemisphere sampling —
  Pharr, Jakob & Humphreys, *Physically Based Rendering* (pbrt).
- The **Cornell box** — the Cornell Program of Computer Graphics' standard radiosity/GI
  validation scene.
