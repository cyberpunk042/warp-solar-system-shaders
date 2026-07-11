# `warp_shaders.lod`

Quality tiers — the one knob that scales every sample-count cost from CPU to a
high-end GPU. All **host** (plain Python).

```python
import warp_shaders as ws
ws.set_active("high")          # process-wide tier
tier = ws.active_tier()        # -> QualityTier
qual = ws.make_quality(tier)   # pack into the Quality uniform for a kernel
```

## `QualityTier`

A dataclass bundling the costs that dominate procedural rendering:

| Field | Meaning |
|---|---|
| `name` | `"low"` / `"medium"` / `"high"` / `"ultra"` |
| `raymarch_steps` | max sphere-tracing steps |
| `shadow_steps` | soft-shadow march steps |
| `ao_steps` | ambient-occlusion taps |
| `noise_octaves` | fBm octaves |
| `volumetric_steps` | cloud/smoke march steps |
| `lut_size` | atmosphere/LUT resolution (per axis) |
| `resolution_scale` | render-resolution multiplier (`1.0` = full) |
| `mip_bias` | texture LOD bias (`+` = blurrier/cheaper) |

The four presets (`TIERS`):

| tier | raymarch | shadow | AO | octaves | volumetric | LUT | res scale | mip bias |
|---|---|---|---|---|---|---|---|---|
| low | 48 | 8 | 3 | 4 | 24 | 32 | 0.75 | 1.0 |
| medium | 96 | 16 | 5 | 5 | 48 | 64 | 1.0 | 0.5 |
| high | 160 | 24 | 8 | 6 | 96 | 128 | 1.0 | 0.0 |
| ultra | 256 | 40 | 12 | 8 | 160 | 256 | 1.0 | 0.0 |

## Functions

| Function | Signature | Notes |
|---|---|---|
| `get_tier` | `(name: str) -> QualityTier` | look up a preset by name |
| `auto_tier` | `(device="cpu") -> str` | pick a tier name for a device (CPU → `low`; CUDA → `high`/`ultra` by VRAM) |
| `set_active` | `(name: str, device="cpu") -> QualityTier` | set the process-wide tier (`"auto"` resolves via `auto_tier`) |
| `active_tier` | `() -> QualityTier` | the current process-wide tier |

A LOD-aware scene reads `active_tier()` in its renderer and threads the counts
into its kernel — see [Concepts → Quality tiers](../concepts.md#quality-tiers-one-knob).
