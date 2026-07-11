# `warp_shaders.textures`

Portable sampling of image maps, baked LUTs, and 3D noise volumes — **hand-
filtered over a plain `wp.array`** so it behaves identically on CPU and CUDA (see
[Concepts → Portability](../concepts.md#portability-arrays-over-hardware-textures)).

```python
from warp_shaders.textures import sample2d, sample3d, sample_equirect   # device
from warp_shaders.textures import to_texture, load_equirect             # host
```

## Device samplers

| Function | Signature | Notes |
|---|---|---|
| `sample2d` | `(tex: array2d(vec3), u: float, v: float, wrap_x: int, wrap_y: int) -> vec3` | bilinear at `uv ∈ [0, 1]`; `wrap_* = 1` repeats, `0` clamps to edge |
| `sample3d` | `(vol: array3d(float), u: float, v: float, w: float, wrap: int) -> float` | trilinear at `(u, v, w) ∈ [0, 1]`; axis order `vol[z, y, x]`; `wrap=1` repeats, `0` clamps |
| `sample_equirect` | `(tex: array2d(vec3), dir: vec3) -> vec3` | sample an equirectangular map by a world-space direction (longitude/latitude) |

## Host loaders

| Function | Signature | Notes |
|---|---|---|
| `to_texture` | `(arr, device="cpu") -> wp.array` | upload a NumPy `(H, W, 3)` (or `(D, H, W)`) array as a sampleable `wp.array` |
| `load_equirect` | `(path, device="cpu", srgb_to_linear=True) -> wp.array2d(vec3)` | load an image (PIL) as an equirect map, optionally converting sRGB → linear |

## Typical uses

- **Image maps** — `load_equirect("earth.png")` on the host, then
  `sample_equirect(tex, surface_dir)` in the kernel (the `earth_map` scene).
- **Baked LUTs** — `build_transmittance_lut()` returns a `wp.array2d(vec3)`
  sampled by `transmittance_lut` (see [engine → atmosphere](engine.md#atmosphere-engineatmosphere)).
- **3D noise volumes** — bake a seamless volume once with `value_tiled3`,
  `to_texture(...)` it, then `sample3d(vol, u, v, w, 1)` for cheap high-frequency
  detail (the `nebula` scene and cloud erosion).
