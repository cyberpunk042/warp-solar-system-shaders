# Research ledger — textures, maps, and LUTs

## Decision: portable array-based sampling (not hardware `wp.Texture`)
Warp 1.15.0 exposes `wp.Texture2D` but **not** `wp.texture_sample` — the
hardware-texture sampling path (CUDA texture units) is incomplete/GPU-only here and
cannot be exercised on this CPU environment. For a cross-tier engine (CPU dev/CI →
GPU) the right choice is **`wp.array2d` + a manual bilinear sampler** written as a
`@wp.func`: identical results on CPU and CUDA, no version/device gating. Hardware
`wp.Texture` remains a future GPU-only acceleration for the same data.

Implication: albedo maps, precomputed LUTs, and 3D noise volumes are all just
`wp.array` inputs sampled with our own `sample2d` / `sample_equirect` / `sample3d`.

## Equirectangular maps
Map a surface direction `n` to UV: `u = atan2(n.z, n.x)/2π + 0.5`,
`v = 0.5 - asin(n.y)/π` (top row = north pole). Bilinear with **longitude wrap**
(x) and **latitude clamp** (y). Baking the Earth albedo once (with the procedural
toolkit at high octave count) and sampling it per frame decouples geographic detail
from per-frame cost — and a real NASA **Blue Marble** equirectangular JPG drops into
the same sampler for photoreal geography (`textures.load_equirect`).

## Transmittance LUT (P10)
Atmospheric transmittance T(h, μ) — altitude h and sun-zenith cosine μ — is a smooth
2D function. Precompute it once into a `wp.array2d` by integrating optical depth to
the atmosphere top, then replace the inner sun light-march with a bilinear lookup.
Removes the O(view×light) nested loop → far cheaper high/ultra sky.
Sources: Bruneton & Neyret 2008; Hillaire 2020 (transmittance + multiscatter LUTs).
