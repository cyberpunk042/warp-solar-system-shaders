# `warp_shaders.engine`

The shading half of the engine: uniforms, PBR, materials, atmosphere,
volumetrics, and the host post pipeline. Import subsystems from the namespace:

```python
from warp_shaders.engine import post                      # host
from warp_shaders.engine.pbr import shade_pbr             # device
from warp_shaders.engine.uniforms import Camera, make_camera, camera_ray_dir
from warp_shaders.engine.material import Material, make_mat, make_material
```

---

## Uniforms — `engine.uniforms`

`@wp.struct` blocks passed as single kernel arguments, plus their host builders.

### Structs

**`Camera`** — `eye: vec3`, `forward: vec3`, `right: vec3`, `up: vec3`,
`tan_half_fov: float`, `aspect: float`.

**`Light`** — `dir: vec3` (toward the light), `color: vec3`, `intensity: float`.

**`Frame`** — `time: float`, `width: int`, `height: int`.

**`Quality`** — `raymarch_steps: int`, `shadow_steps: int`, `ao_steps: int`,
`noise_octaves: int`, `volumetric_steps: int`, `mip_bias: float`. Built from a
[`QualityTier`](lod.md).

### Device function

`camera_ray_dir(cam: Camera, u: float, v: float) -> vec3` — the normalized
primary ray direction for normalized screen coords `u, v ∈ [-1, 1]`. This is the
one call every render kernel starts with.

### Host builders

| Function | Signature |
|---|---|
| `make_camera` | `(eye, target, fov_deg=45.0, aspect=1.0, up=(0,1,0)) -> Camera` |
| `make_light` | `(direction, color=(1,1,1), intensity=1.0) -> Light` |
| `make_frame` | `(time, width, height) -> Frame` |
| `make_quality` | `(tier: QualityTier) -> Quality` |

---

## Material — `engine.material`

A typed surface so scenes describe a material once and shade it uniformly.

**`Material`** (`@wp.struct`) — `albedo: vec3`, `roughness: float`,
`metallic: float`, `emission: vec3`.

| Function | Where | Signature |
|---|---|---|
| `shade_material` | device | `(m: Material, n, v, l, light_color, light_intensity) -> vec3` — direct PBR radiance from one light plus the material's self-emission |
| `make_mat` | device | `(albedo: vec3, roughness: float, metallic: float) -> Material` — construct a non-emissive material inside a kernel |
| `make_material` | host | `(albedo, roughness=0.5, metallic=0.0, emission=(0,0,0)) -> Material` |

`n`, `v`, `l` are unit vectors: surface normal, direction to the eye, direction
to the light.

---

## PBR — `engine.pbr`

GGX Cook-Torrance building blocks (all **device**). Use `shade_material` for the
common path; reach for these when you need the terms directly.

| Function | Signature | Role |
|---|---|---|
| `shade_pbr` | `(n, v, l, albedo, rough, metallic, light_color) -> vec3` | full direct radiance from one light |
| `fresnel_schlick` | `(cos_theta: float, f0: vec3) -> vec3` | Fresnel reflectance |
| `distribution_ggx` | `(n_dot_h: float, rough: float) -> float` | GGX normal-distribution term |
| `geometry_schlick_ggx` | `(n_dot_x: float, rough: float) -> float` | single-direction geometry term |
| `geometry_smith` | `(n_dot_v: float, n_dot_l: float, rough: float) -> float` | Smith masking-shadowing |

---

## Shading helpers — `engine.shading`

Small, map-independent **device** helpers every scene tends to re-derive. Compose
the look instead of copy-pasting it.

| Function | Signature | Role |
|---|---|---|
| `apply_fog` | `(col: vec3, dist: float, fog_col: vec3, density: float) -> vec3` | exponential distance fog / aerial perspective — `f = 1 - exp(-density·dist)` |
| `sun_disk` | `(rd: vec3, sun: vec3, disk_col: vec3, size: float, glow: float) -> vec3` | additive sun disk + soft halo for a background ray (`size ≈ 0.9990–0.9999`) |
| `sky_gradient` | `(rd: vec3, horizon: vec3, zenith: vec3) -> vec3` | two-stop vertical sky gradient by ray elevation |

---

## Atmosphere — `engine.atmosphere`

Physically based single-scatter sky (Rayleigh + Mie, Cornette–Shanks phase,
ground shadowing), in analytic and LUT-accelerated forms.

### Analytic (device)

| Function | Signature |
|---|---|
| `atmosphere` | `(ro, rd, sun, view_samples: int, light_samples: int) -> vec3` — in-scattered radiance along a view ray; `ro` in planet-centred metres |
| `sky_radiance` | `(ro, rd, sun, view_samples, light_samples) -> vec3` — `atmosphere` plus the sun disk |

### LUT-accelerated

| Function | Where | Signature |
|---|---|---|
| `build_transmittance_lut` | host | `(size=64, device="cpu", steps=32) -> wp.array2d(vec3)` — bake the transmittance table once |
| `transmittance_lut` | device | `(lut, h: float, mu: float) -> vec3` — sample the table |
| `atmosphere_lut` | device | `(ro, rd, sun, view_samples: int, lut) -> vec3` — sky in-scatter using the LUT for the sun path (no inner loop) |
| `sky_radiance_lut` | device | `(ro, rd, sun, view_samples, lut) -> vec3` — `atmosphere_lut` plus the sun disk |
| `sample_counts` | host | `(tier_name: str) -> (view_samples, light_samples)` — per-tier sample budget |

Prefer the LUT path in tier-scaled scenes: build the LUT once on the host at
`active_tier().lut_size`, then pass it into the kernel.

---

## Volumetrics — `engine.volumetric`

Cloud density and light-marching (Schneider density, Henyey–Greenstein phase,
Beer–Lambert extinction). All **device**.

| Function | Signature |
|---|---|
| `hg_phase` | `(cos_theta: float, g: float) -> float` — Henyey–Greenstein phase |
| `cloud_density` | `(p: vec3, time: float, coverage: float, base_y: float, top_y: float) -> float` — density in `[0, 1]` inside the slab `[base_y, top_y]` |
| `march_clouds` | `(ro, rd, sun, time, coverage, base_y, top_y, steps: int, light_steps: int, sun_col, amb) -> vec4` — raymarch a horizontal cloud slab; returns `(scattered_rgb, transmittance)` |

`steps`/`light_steps` come from the tier (`volumetric_steps`), so cloud detail
scales with quality.

---

## Post — `engine.post`

Host-side (NumPy) tonemap, bloom, godrays, and vignette. Operate on the HDR
`(H, W, 3)` array you pull back with `img.numpy()`, in this order:
bloom/godrays → tonemap → vignette.

| Function | Signature |
|---|---|
| `tonemap` | `(frame, mode="aces", exposure=1.0, gamma=2.2) -> ndarray` — `mode` ∈ `aces` / `agx` / `reinhard`; maps HDR → `[0, 1]` |
| `bloom` | `(hdr, threshold=1.0, strength=0.6, radius=6, passes=3) -> ndarray` — bright-pass + blur bleed |
| `godrays` | `(hdr, cx, cy, samples=28, density=0.9, decay=0.95, weight=0.6, ...) -> ndarray` — radial light scattering from screen point `(cx, cy)` |
| `vignette` | `(frame, amount=0.35) -> ndarray` — darken toward the corners |

> **Sources.** PBR: Cook–Torrance GGX (Karis/UE4). Atmosphere:
> Nishita/O'Neil single-scatter with a Bruneton/Hillaire transmittance LUT.
> Volumetrics: Schneider "Nubis", Henyey–Greenstein, Beer–Lambert. Post: ACES
> (Narkowicz), AgX (Benjamin), GPU Gems 3 radial godrays. Full citations in
> [`docs/research/`](../research/00-foundations.md).
