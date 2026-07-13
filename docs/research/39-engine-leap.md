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

## Beyond diffuse — specular, subsurface, and time

Once rays bounce, every material is just a rule for *how a ray leaves a surface*, and the same
integrator absorbs them all:

- **Specular (mirror + glass).** A mirror reflects the incoming ray about the normal; a
  dielectric splits into a reflected and a refracted ray by **Snell's law**, with the split
  ratio given by the **Fresnel** term (Schlick's approximation) and **total internal
  reflection** when the refraction has no real solution. Choosing one branch per bounce
  (Russian-roulette on the Fresnel probability) keeps the estimator unbiased. This is
  reflection *and* refraction sharing one path tracer with the diffuse walls.
- **Subsurface scattering.** Wax, jade, marble and skin are neither opaque nor clear: light
  enters, **random-walks inside** the medium — exponential free-flight steps, isotropic
  scatter, a single-scattering albedo multiplied in at each step — and leaves elsewhere.
  Thin regions let the walk escape and **glow**; thick regions absorb it. Modelled here as a
  literal bounded random walk against the object's own SDF, so it is the volumetric analogue
  of the surface path trace.
- **Motion blur.** A real shutter is open for a slice of *time*. Because path tracing already
  averages many samples per pixel, each sample can also pick a **random instant within the
  shutter** and evaluate the geometry at that instant — moving objects smear, static ones stay
  sharp, for free. Temporal sampling is spatial anti-aliasing extended by one dimension.

## Scenes

`cornell_box` — the canonical global-illumination test: a five-wall open-front room (red left,
green right, white rest) with two white blocks, lit only by a ceiling emitter, path-traced so
the coloured walls bleed onto the blocks and floor with soft contact shadows. The "hello
world" of GI, and the proof the path tracer works.

`glass_box` — the same room with **specular materials**: a glass sphere that refracts an
inverted image of the room and glints by Fresnel, and a mirror sphere that reflects the red/
green walls and the ceiling light. Reflection, refraction and diffuse GI share one unbiased
integrator.

`subsurface` — a jade sphere and a thin standing ring backlit by a warm panel, with light
entering the translucent medium and **random-walking** until it escapes. The thin ring glows
right through; the sphere shows a bright translucent rim around a denser, warmer core.

`motion_blur` — three spheres translating left-to-right at rising speeds (sharp → smear → long
streak) and a striped sphere whose spin blurs its bands, all from **distributed temporal
sampling** in the same integrator that jitters rays for anti-aliasing.

*(more to come as the strand grows: caustics focusing, and re-renders of flagship scenes at the
new fidelity tier.)*

## Sources

- The **rendering equation** — Kajiya, *The Rendering Equation* (SIGGRAPH 1986).
- Monte-Carlo path tracing, importance sampling, cosine-weighted hemisphere sampling —
  Pharr, Jakob & Humphreys, *Physically Based Rendering* (pbrt).
- The **Cornell box** — the Cornell Program of Computer Graphics' standard radiosity/GI
  validation scene.
