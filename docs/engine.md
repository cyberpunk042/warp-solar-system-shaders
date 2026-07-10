# The Warp rendering engine

A reusable, tiered, procedural rendering engine built on NVIDIA Warp. Everything
is `@wp.func` device code (composable into your own `@wp.kernel`) plus a small
host layer for uniforms, LOD, and post. Research notes + citations live in
[`research/`](research/).

## Module map

```
warp_shaders/
  procedural/           # building blocks
    hash.py             hash11/21/22/31/33            (McGuire)
    noise.py            value3, noised3 (value + analytic gradient), perlin3,
                        worley3 / worley3_f2, fbm3, fbm_perlin3, ridged3, billow3,
                        domain_warp3, curl3           (IQ, Gustavson, Worley, Bridson)
    sdf.py              sd_sphere/box/round_box/torus/cylinder/capsule/plane/ellipsoid,
                        op_union/intersect/subtract/smooth_*/round/onion   (IQ)
  engine/
    uniforms.py         @wp.struct Camera/Light/Frame/Quality + make_* builders,
                        camera_ray_dir(cam, u, v)
    pbr.py              fresnel_schlick, distribution_ggx, geometry_smith,
                        shade_pbr(n, v, l, albedo, rough, metallic, light_color)
    atmosphere.py       atmosphere(ro, rd, sun, view_samples, light_samples),
                        sky_radiance(...), sample_counts(tier)   (Nishita/O'Neil)
    volumetric.py       hg_phase, cloud_density, march_clouds(...)   (Schneider/HG/Beer)
    post.py             tonemap(mode=aces|reinhard|agx), bloom, vignette   (host/NumPy)
  lod.py                QualityTier presets, get_tier/auto_tier/set_active/active_tier
```

## Quality tiers

One knob scales the sample-count / octave costs that dominate procedural
rendering. `render.py --quality {auto,low,medium,high,ultra}` sets the process
tier; `auto` picks by device (CPU → low, CUDA → high/ultra by VRAM).

| tier | raymarch | shadow | AO | octaves | volumetric | LUT | res |
|---|---|---|---|---|---|---|---|
| low | 48 | 8 | 3 | 4 | 24 | 32 | 0.75 |
| medium | 96 | 16 | 5 | 5 | 48 | 64 | 1.0 |
| high | 160 | 24 | 8 | 6 | 96 | 128 | 1.0 |
| ultra | 256 | 40 | 12 | 8 | 160 | 256 | 1.0 |

A LOD-aware scene reads `warp_shaders.lod.active_tier()` in its renderer and
passes the counts into its kernel (see `scenes/sky.py`, `scenes/earth_v2.py`).

## Writing an engine scene

Two shapes are supported by the `Scene` contract:

1. **Kernel scene** — a fixed `(img, width, height, time, mouse)` kernel; set
   `Scene(name=..., kernel=..., description=...)`. Good for self-contained shaders.
2. **Renderer scene** — a `renderer(width, height, time, mouse, device)` callback
   returning an `(H, W, 3)` array; set `Scene(name=..., renderer=..., ...)`. Use
   this when you need uniform structs, the LOD tier, or host post. Pattern:

```python
def _render(width, height, time, mouse, device):
    tier = active_tier()
    cam  = make_camera(eye, target, fov_deg=45, aspect=width / height)
    light = make_light(sun_dir); qual = make_quality(tier); frame = make_frame(time, width, height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, light, qual, frame], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=6)
    return post.tonemap(hdr, mode="aces", exposure=1.1)
```

`scenes/pbr_demo.py` is the canonical copy-me template (adaptive raymarch +
gradient normals + soft shadow + AO + PBR + post).

## Gotchas (Warp specifics)

- **No `vec * vec` operator** — component-wise multiply is `wp.cw_mul(a, b)`.
- **No function pointers** — a raymarcher inlines its scene `map()`; reusable
  pieces are the map-independent funcs (PBR, atmosphere, post, noise, SDF).
- Dynamic `for _ in range(n)` loops (runtime `n`) are supported — that's how the
  LOD sample counts flow into kernels.
- Hardware textures (`wp.Texture2D/3D`, v1.12+) enable precomputed atmosphere LUTs
  and image maps — the next realism tier (see `research/00-foundations.md`).
