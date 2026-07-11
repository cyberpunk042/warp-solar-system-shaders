# `warp_shaders.engine`

The shading half of the engine: uniforms, PBR, materials, atmosphere,
volumetrics, colour science, ray intersection, sky backgrounds, analytic
shadows/AO, the host post pipeline (with named looks) and HDR frame output.
Import subsystems from the namespace:

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
`tan_half_fov: float`, `aspect: float`, `aperture: float` (lens radius; `0` =
pinhole), `focus_dist: float` (distance to the sharp focal plane).

**`Light`** — `dir: vec3` (toward the light), `color: vec3`, `intensity: float`.

**`Frame`** — `time: float`, `width: int`, `height: int`.

**`Quality`** — `raymarch_steps: int`, `shadow_steps: int`, `ao_steps: int`,
`noise_octaves: int`, `volumetric_steps: int`, `mip_bias: float`. Built from a
[`QualityTier`](lod.md).

### Device function

`camera_ray_dir(cam: Camera, u: float, v: float) -> vec3` — the normalized
primary ray direction for normalized screen coords `u, v ∈ [-1, 1]`. This is the
one call every render kernel starts with.

### Depth of field (thin lens)

By default the camera is a pinhole (`aperture = 0`) and every existing scene is
unaffected. Pass `aperture` (and optionally `focus_dist`) to `make_camera` for a
thin-lens camera, then accumulate `K` lens samples per pixel in your kernel:

| Function | Signature | Role |
|---|---|---|
| `lens_offset` | `(cam, s1: float, s2: float) -> vec3` | world-space aperture offset from the eye for lens samples `s1,s2 ∈ [0,1]` |
| `focus_point` | `(cam, u: float, v: float) -> vec3` | the on-focal-plane point for pixel `(u,v)` — objects here stay sharp |

```python
# inside the kernel, per pixel:
fp = focus_point(cam, u, v)
acc = wp.vec3(0.0, 0.0, 0.0)
for k in range(dof_samples):
    off = lens_offset(cam, hash(...), hash(...))
    ro = cam.eye + off
    rd = wp.normalize(fp - ro)
    acc += shade(ro, rd)        # raymarch this jittered ray
acc = acc / float(dof_samples)
```

### Host builders

| Function | Signature |
|---|---|
| `make_camera` | `(eye, target, fov_deg=45.0, aspect=1.0, up=(0,1,0), aperture=0.0, focus_dist=None) -> Camera` — `focus_dist=None` focuses on the look-at target |
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

### LUT-accelerated (single + multiple scattering)

Two precomputed 2D LUTs, both indexed `[altitude, sun-zenith mu]`: a
**transmittance** LUT (removes the per-view-sample sun light-march) and a
**Hillaire 2020 multiple-scattering** LUT (adds the isotropic multi-scatter that
brightens twilight / horizon / shadowed sky without touching the single-scatter
term). The multiscatter LUT consumes the transmittance LUT and is baked once on
the same device.

| Function | Where | Signature |
|---|---|---|
| `build_transmittance_lut` | host | `(size=64, device="cpu", steps=32) -> wp.array2d(vec3)` — bake the transmittance table once |
| `transmittance_lut` | device | `(lut, h: float, mu: float) -> vec3` — sample the transmittance table |
| `build_multiscatter_lut` | host | `(tr_lut, size=32, device="cpu", dir_samples=32, steps=32) -> wp.array2d(vec3)` — bake the multiple-scattering table (needs `tr_lut`) |
| `multiscatter_lut` | device | `(ms_lut, h: float, mu: float) -> vec3` — sample the multiscatter table |
| `atmosphere_lut` | device | `(ro, rd, sun, view_samples: int, lut, ms_lut) -> vec3` — single + multiple scattering via the LUTs (no inner loop) |
| `sky_radiance_lut` | device | `(ro, rd, sun, view_samples, lut, ms_lut) -> vec3` — `atmosphere_lut` plus the sun disk |
| `sample_counts` | host | `(tier_name: str) -> (view_samples, light_samples)` — per-tier sample budget |

Prefer the LUT path in tier-scaled scenes: build **both** LUTs once on the host
at `active_tier().lut_size` (`tr = build_transmittance_lut(...)` then
`ms = build_multiscatter_lut(tr, ...)`), cache them, and pass both into the
kernel. See `scenes/sky.py` for the canonical wiring.

---

## Volumetrics — `engine.volumetric`

Cloud density and light-marching (Schneider density, Henyey–Greenstein phase,
Beer–Lambert extinction). The high-frequency edge erosion is read from a
**baked, seamless 3D detail volume** (`build_cloud_detail`) with trilinear
`sample3d` — much cheaper than recomputing Worley/fBm per march step (~45 %
faster on the `clouds` scene) and seamless via `value_tiled3`.

| Function | Where | Signature |
|---|---|---|
| `build_cloud_detail` | host | `(size=96, device="cpu") -> wp.array3d(float)` — bake the seamless detail volume once |
| `hg_phase` | device | `(cos_theta: float, g: float) -> float` — Henyey–Greenstein phase |
| `cloud_density` | device | `(p: vec3, time, coverage, base_y, top_y, vol) -> float` — density in `[0, 1]` inside the slab; `vol` from `build_cloud_detail` |
| `march_clouds` | device | `(ro, rd, sun, time, coverage, base_y, top_y, steps: int, light_steps: int, sun_col, amb, vol) -> vec4` — raymarch a horizontal cloud slab; returns `(scattered_rgb, transmittance)` |

`steps`/`light_steps` come from the tier (`volumetric_steps`), so cloud detail
scales with quality.

---

## Colour — `engine.color`

Blackbody temperature, sRGB transfer, luminance — device `@wp.func`s + host
`*_np` twins. Consolidates the temperature-colour code across the star shaders,
particle ramp and lava.

| Function | Signature |
|---|---|
| `kelvin_to_rgb` | `(kelvin) -> vec3` — blackbody colour on the Planckian locus (Tanner-Helland fit), peak-normalised |
| `blackbody` | `(t) -> vec3` — artistic 0..1 heat ramp (red → white → blue) |
| `luminance` | `(c) -> float` — Rec.709 relative luminance |
| `linear_to_srgb` / `srgb_to_linear` | `(c) -> vec3` — gamma transfer |
| host | `kelvin_to_rgb_np`, `linear_to_srgb_np`, `srgb_to_linear_np`, `luminance_np` |

## Intersection — `engine.intersect`

The shared analytic ray tests (miss on a sphere/box returns `(1e30, -1e30)`).

| Function | Signature |
|---|---|
| `ray_sphere` / `ray_sphere_o` | `(ro, rd, [center,] radius) -> vec2` — (t_near, t_far) |
| `sphere_t` | `(ro, rd, center, radius) -> float` — nearest positive t, or -1 |
| `ray_plane` / `ray_disk` | `(ro, rd, p0/center, n[, radius]) -> float` — t, or -1 |
| `ray_box` | `(ro, rd, bmin, bmax) -> vec2` — AABB slab (t_near, t_far) |

## Sky — `engine.sky`

| Function | Signature |
|---|---|
| `starfield` | `(rd) -> vec3` — two star-size layers + colour temperature across the sky |
| `milky_way` | `(rd, axis, intensity) -> vec3` — fBm galactic-plane glow band |

## Post — `engine.post`

Host-side (NumPy) pipeline over the HDR `(H, W, 3)` array you pull back with
`img.numpy()`. Recommended order: `exposure/auto_exposure` → `bloom/godrays` →
`tonemap` → `chromatic_aberration` → `sharpen` → `vignette` → `film_grain`.

| Function | Signature |
|---|---|
| `exposure` / `auto_exposure` | `(hdr, ev=0.0)` / `(hdr, key=0.18, max_gain=8.0)` — photographic stops / geometric-mean-luminance auto key (before tonemap) |
| `tonemap` | `(frame, mode="aces", exposure=1.0, gamma=2.2) -> ndarray` — `mode` ∈ `aces` / `agx` / `reinhard`; maps HDR → `[0, 1]` |
| `bloom` | `(hdr, threshold=1.0, strength=0.6, radius=6, passes=3) -> ndarray` — bright-pass + blur bleed |
| `godrays` | `(hdr, cx, cy, samples=28, density=0.9, decay=0.95, weight=0.6, ...) -> ndarray` — radial light scattering from screen point `(cx, cy)` |
| `chromatic_aberration` | `(frame, amount=0.004) -> ndarray` — radial red-out / blue-in lens dispersion |
| `sharpen` | `(frame, amount=0.5, radius=2) -> ndarray` — unsharp mask |
| `vignette` | `(frame, amount=0.35) -> ndarray` — darken toward the corners |
| `film_grain` | `(frame, amount=0.04, seed=0) -> ndarray` — deterministic filmic grain |

### Named looks

One-call display-range grades composed from the ops above. Apply after tonemap.

| Function | Signature |
|---|---|
| `looks` | `() -> list[str]` — the preset names: `clean`, `cinematic`, `film`, `dreamy`, `crisp` |
| `apply_look` | `(frame, look="clean", seed=0) -> ndarray` — grade a `[0,1]` frame with a preset **name** or a raw params dict (`glow` / `ca` / `sharpen` / `vignette` / `grain`) |

`render.py --look <name>` applies one per frame; `ws.render_image(scene, look=...)`
renders a scene and grades it in one call.

## Shadows & AO — `engine.shadow`

Reusable, self-contained analytic soft shadows + ambient occlusion (**device**) —
no SDF callback, so they compile into any kernel (planets, moons, the
`shadow_demo` scene). See `scenes/shadow_demo.py`.

| Function | Signature | Role |
|---|---|---|
| `soft_shadow_sphere` | `(p, l, ce, ra, k) -> float` | penumbra soft shadow of a sphere occluder along light dir `l`; `1`=lit, `0`=shadow, hardness `k` |
| `sphere_occlusion` | `(p, n, ce, ra) -> float` | exact analytic occluded fraction of the hemisphere by a sphere |
| `sphere_ao` | `(p, n, ce, ra) -> float` | occlusion **visibility** (`1 - sphere_occlusion`), multiplies in like a shadow |
| `penumbra` | `(res, h, t, k) -> float` | the IQ SDF soft-shadow running-min atom, `res = min(res, k·h/t)` |
| `ground_contact_ao` | `(height, radius) -> float` | cheap contact darkening for a subject above a plane |

## Reflection & refraction — `engine.raytrace`

The device atoms a raymarcher needs to **bounce** a ray at a hit (the multi-bounce
loop lives in the scene — Warp has no recursion; see `scenes/reflections.py`).
Conventions: `i` = incident direction (unit, toward the surface), `n` = outward
normal, `eta` = `n1/n2` across the interface.

| Function | Signature | Role |
|---|---|---|
| `reflect` | `(i, n) -> vec3` | mirror reflection |
| `refract` | `(i, n, eta) -> vec3` | Snell refraction; total internal reflection falls back to `reflect` |
| `refract_k` | `(i, n, eta) -> float` | the Snell discriminant; `< 0` ⇒ TIR |
| `schlick_f0` | `(ior) -> float` | normal-incidence reflectance `F0` of a dielectric |
| `fresnel_dielectric` | `(cos_theta, ior) -> float` | Schlick Fresnel of a dielectric (`F0` head-on → `1` grazing) |
| `fresnel_schlick_s` | `(cos_theta, f0) -> float` | scalar Schlick from an explicit `F0` (metals/tinted mirrors) |

## Frame output — `engine.imageio`

LDR + true-HDR containers and a `RenderTarget` wrapper (**host**). Scenes render a
linear buffer that often exceeds display range (stars, suns, bloom); PNG discards
it, these keep it.

| Function / class | Signature | Role |
|---|---|---|
| `save_png` | `(path, frame)` | clamp `[0,1]` and write 8-bit (no tonemap) |
| `save_npy` | `(path, frame)` | lossless raw float32 `.npy` |
| `save_hdr` / `load_hdr` | `(path, frame)` / `(path) -> ndarray` | Radiance **RGBE** `.hdr` — shared 8-bit exponent, any compositor reads it, pure-NumPy |
| `RenderTarget` | `RenderTarget(hdr).save(path)` | dispatch by extension (`.png` tonemaps, `.npy`/`.hdr` keep linear); `.tonemapped(mode, exposure)` |

`render.py -o out.hdr` (or `.npy`) writes the raw linear buffer instead of a PNG.

## Video — `engine.video`

Encode a frame sequence to an animated container (**host**); the extension picks
the format. See the [Cinematics guide](../guides/cinematics.md).

| Function | Signature | Role |
|---|---|---|
| `write_video` | `(frames, path, fps=30, quality=8) -> str` | `.mp4`/`.webm` via imageio-ffmpeg when installed (else an animated `.webp` beside the path); `.webp`/`.gif`/`.apng` via Pillow. Returns the path written |
| `crossfade` | `(a, b, steps) -> iter` | yield `steps` frames dissolving `a` into `b` (the reel's transition) |
| `save_frames` | `(frames, out_dir, prefix="frame") -> int` | write a numbered PNG per frame |

`render.py --video PATH` uses this per render.

## Camera paths — `engine.camera_path`

Keyframed camera motion (**host**, NumPy). A `CameraPath` interpolates the **eye**
with a Catmull-Rom spline through the keyframes and eases **target**/**FOV**
between them; `camera(t, aspect)` builds the engine `Camera`.

| Symbol | Signature | Role |
|---|---|---|
| `Keyframe` | `(t, eye, target=(0,0,0), fov=45.0)` | one stop on the path |
| `CameraPath` | `CameraPath(keyframes, easing="ease_in_out")` | `.sample(t) -> (eye, target, fov)`, `.camera(t, aspect)`, `.add(...)` |
| `orbit` | `(center, radius, elevation, turns=1.0, fov=45, ...) -> CameraPath` | circle the subject |
| `dolly` | `(eye0, eye1, target, fov0, fov1=None, ...) -> CameraPath` | push-in / pull-out |
| `fly` | `(keyframes, easing=...) -> CameraPath` | from `(t, eye[, target, fov])` tuples |
| `EASINGS` / `ease` | `ease(name, t)` | `linear` / `smoothstep` / `smoother` / `ease_in` / `ease_out` / `ease_in_out` |

The cosmos `render_system(..., camera=(eye, target, fov))` accepts a sampled path;
the `ss_flyby` scene wires a looping orbit around the trinary system.

> **Sources.** PBR: Cook–Torrance GGX (Karis/UE4). Atmosphere:
> Nishita/O'Neil single-scatter with a Bruneton/Hillaire transmittance LUT.
> Volumetrics: Schneider "Nubis", Henyey–Greenstein, Beer–Lambert. Post: ACES
> (Narkowicz), AgX (Benjamin), GPU Gems 3 radial godrays. Full citations in
> [`docs/research/`](../research/00-foundations.md).
