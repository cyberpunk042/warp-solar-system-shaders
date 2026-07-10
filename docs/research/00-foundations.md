# Research ledger — foundations

> Research-first rule: every technique we implement is recorded here with a
> primary source and a Warp-implementation note before/at the time it lands.
> Each device function also cites its source in-code.

## NVIDIA Warp — rendering-relevant capabilities
- **Hardware textures (v1.12+)**: `wp.Texture1D/2D/3D`, `FILTER_POINT|LINEAR`
  (bi/tri-linear), `ADDRESS_{WRAP,CLAMP,MIRROR,BORDER}`, **mipmaps** (`num_mip_levels`,
  `mip_filter_mode`) with an optional `lod` arg to `wp.texture_sample`, and
  `copy_from_array`/`copy_to_array`. → precomputed atmosphere LUTs, 3D cloud-noise
  volumes, and image maps (Blue Marble) are first-class; hardware mip-LOD maps onto
  our quality tiers.
- **`@wp.struct`**: lets us pass Camera/Light/Frame/Quality "uniform" blocks to
  kernels (the UBO pattern from the-virus-block-mc). Clean-API backbone.
- **No rasterizer**: rendering is kernel-side raymarch/analytic + splatting. Fits a
  procedural engine; per-pixel kernels write an `array2d(dtype=wp.vec3)` HDR buffer.
- Source: NVIDIA/warp CHANGELOG + docs (github.com/NVIDIA/warp).

## Procedural / noise — primary sources (implemented in warp_shaders/procedural)
- **Hashes**: Morgan McGuire, "Hash Functions for GPU Rendering" (JCGT 2020) — sin-free
  integer-ish hashes; we use the compact `fract(sin(dot()))` family for portability
  (CPU+CUDA), documented as such.
- **Value / gradient (Perlin) noise + analytic derivatives**: Inigo Quilez,
  "value noise derivatives" and "gradient noise derivatives" (iquilezles.org/articles/
  morenoise, /gradientnoise) — quintic interpolation, returns value + gradient so
  normals need no finite differences.
- **Simplex noise**: Ken Perlin / Stefan Gustavson, "Simplex noise demystified" (2005).
- **Worley / cellular**: Steven Worley, "A Cellular Texture Basis Function" (1996);
  F1/F2 over a 3×3×3 cell neighborhood.
- **fbm / domain warping**: Inigo Quilez, "fBM" and "domain warping"
  (iquilezles.org/articles/fbm, /warp). Ridged/billow are |n| variants.
- **Curl noise** (divergence-free flow): Bridson et al., "Curl-Noise for Procedural
  Fluid Flow" (SIGGRAPH 2007) — curl of a noise potential.

## LOD / quality tiers
One `Quality` knob (low/med/high/ultra) scales: raymarch max-steps, noise octaves,
shadow/AO steps, volumetric steps, LUT sizes, resolution scale, texture mip bias.
Default tier auto-detected from the active device (CPU → low; CUDA → high/ultra by VRAM).
Rationale: sample-count and octave scaling are the dominant cost knobs in raymarched
procedural rendering; exposing one enum keeps scenes device-agnostic.

## Deferred (later phases, will get their own note)
Atmosphere (O'Neil GPU Gems 2 · Bruneton-Neyret 2008 · Hillaire 2020) — P3.
PBR (Cook-Torrance/GGX, Walter 2007 · Schlick · Karis UE4) — P2.
Volumetric clouds (Schneider & Vos "Nubis" · Henyey-Greenstein · Beer-Lambert) — P4.
Tonemap (ACES · AgX) — P2.

## Queued for P1b (procedural toolkit round 2)
- `simplex3` (Perlin/Gustavson simplex) — fewer directional artifacts than value/Perlin.
- Tileable/periodic noise variants (for seamless textures + LUT baking).
- SDF gallery scene; fold legacy `warp_shaders/{sdf,particles,earthgfx}.py` into
  thin re-exports over `procedural/` once P2 consumes them.

