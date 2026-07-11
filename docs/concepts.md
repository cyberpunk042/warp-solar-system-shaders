# Concepts

The mental model behind the engine. Read this once and the API pages will read
themselves.

## Device vs host

Warp Shaders is split into two kinds of code, and knowing which is which is the
single most important thing:

| | **Device** (`@wp.func` / `@wp.kernel`) | **Host** (plain Python) |
|---|---|---|
| Runs on | the GPU/CPU, once per pixel | the interpreter, once per frame |
| Examples | `fbm3`, `sd_sphere`, `shade_pbr`, `atmosphere`, `camera_ray_dir` | `make_camera`, `post.tonemap`, `build_transmittance_lut`, `set_active` |
| You call it | **inside your own `@wp.kernel`** | in your `renderer()` callback |
| Compiled? | yes — JIT to CUDA/CPU | no |

The procedural toolkit and the shading device functions (`ws.procedural`,
`ws.engine.pbr`, `ws.engine.atmosphere`, `ws.engine.volumetric`,
`ws.textures.sample*`) are **device functions**. The uniform builders
(`make_*`), the post pipeline (`ws.post.*`), and the LUT bakers
(`build_*_lut`) are **host** helpers. The `Material` struct has both a device
constructor (`make_mat`, called in a kernel) and a host one (`make_material`).

## The render pipeline

Every LOD-aware scene follows the same shape:

```
active_tier()                 ── pick the quality tier (host)
   │
make_camera / make_light      ── pack uniforms as @wp.struct (host)
make_quality / make_frame
   │
wp.launch(render_kernel, …)   ── one thread per pixel (device)
   │   camera_ray_dir()        ── primary ray from the camera struct
   │   raymarch map()          ── sphere-trace the SDF (or slab-march a volume)
   │   normal / shadow / AO     ── gradient normal, soft shadow, ambient occlusion
   │   shade_material / atmosphere / march_clouds
   │        → writes an HDR vec3 into img[i, j]
   │
img.numpy()                   ── pull the HDR buffer back to the host
post.bloom / godrays          ── optional light bleeding (host, NumPy)
post.tonemap(mode="aces")     ── HDR → display [0, 1]
post.vignette                 ── final framing
```

The kernel produces **linear HDR** radiance; tone-mapping to a displayable
`[0, 1]` image happens on the host, so bloom and godrays operate on true HDR
values before compression.

## Quality tiers (one knob)

Procedural rendering cost is dominated by **sample counts** — raymarch steps,
shadow steps, AO taps, fBm octaves, volumetric steps, LUT resolution. A
[`QualityTier`](api/lod.md) bundles all of them, and `set_active(name)` (or
`--quality` on the CLI) chooses one process-wide:

| tier | raymarch | shadow | AO | octaves | volumetric | LUT | res scale |
|---|---|---|---|---|---|---|---|
| **low** | 48 | 8 | 3 | 4 | 24 | 32 | 0.75 |
| **medium** | 96 | 16 | 5 | 5 | 48 | 64 | 1.0 |
| **high** | 160 | 24 | 8 | 6 | 96 | 128 | 1.0 |
| **ultra** | 256 | 40 | 12 | 8 | 160 | 256 | 1.0 |

A scene reads `active_tier()` in its renderer and threads the counts into its
kernel — usually via `make_quality(tier)`, which packs them into the `Quality`
uniform. Then a `for _ in range(qual.raymarch_steps)` loop scales for free. This
is why the same scene runs on a CPU (`low`/`auto`) and a 4090 (`ultra`):
**nothing but the counts change.** `auto` picks by device (CPU → `low`,
CUDA → `high`/`ultra` by VRAM).

## Portability: arrays over hardware textures

Image maps, baked LUTs, and 3D noise volumes are all sampled through
[`ws.textures`](api/textures.md) — `sample2d`, `sample3d`, `sample_equirect` —
which do **bilinear/trilinear filtering by hand over a plain `wp.array`**.

Warp does expose hardware textures (`wp.Texture2D`), but `wp.texture_sample`
isn't usable on the CPU backend, which would break the "runs everywhere"
promise. Sampling arrays directly costs a little arithmetic but works
identically on CPU and CUDA — so a scene developed on a laptop behaves the same
on the GPU. This is how `earth_map` drops in a NASA equirectangular texture and
how `nebula` reads a baked 3D noise volume.

## Atmosphere: analytic and LUT-accelerated

Physically based sky comes in two forms (see [engine API](api/engine.md)):

- `atmosphere(ro, rd, sun, view_samples, light_samples)` — a full
  single-scatter integral (Rayleigh + Mie, Cornette–Shanks phase, ground
  shadowing). Correct, but a nested loop per pixel.
- `atmosphere_lut(ro, rd, sun, view_samples, lut)` — the same physics with the
  sun-path optical depth read from a **precomputed transmittance LUT**
  (`build_transmittance_lut()` on the host), removing the inner loop. Faster,
  and the LUT resolution scales with the tier.

## The scene registry

[`Scene`](api/scene.md) objects are auto-discovered: `list_scenes()` imports
every module in `warp_shaders/scenes/` and collects each module-level `SCENE`
(or `SCENES`). Adding a scene is adding a file. A `Scene` accepts either:

- a **kernel** — a fixed `(img, width, height, time, mouse)` signature, for
  self-contained shaders; or
- a **renderer** — a `renderer(width, height, time, mouse, device)` callback
  returning an `(H, W, 3)` array, for anything that needs uniforms, the LOD
  tier, or host post.

See [Writing a scene](guides/writing-a-scene.md) for both.

## Warp gotchas (worth knowing up front)

- **No `vec * vec` operator.** Component-wise multiply is `wp.cw_mul(a, b)`.
  `vec * scalar` is fine.
- **No function pointers.** A raymarcher must *inline* its scene `map()` — you
  can't pass a `map` function into a shared marcher. The reusable pieces are the
  map-independent functions (PBR, atmosphere, post, noise, SDF).
- **No `import` inside `@wp.func`.** Imports go at module top level.
- **Dynamic loops are OK.** `for _ in range(runtime_int)` compiles fine — that's
  exactly how tier sample counts flow into kernels.
- **`@wp.struct` for uniform blocks.** Group camera/light/frame/quality into
  structs and pass them as single kernel arguments.
- **Array indexing.** `wp.array2d` is `[i, j]` (row, col); `wp.array3d` is
  `[z, y, x]`.
- **Kernels must be file-defined.** Warp can't compile a kernel built from an
  `exec`'d string, so verify via `render.py`, not inline `python -c`.
