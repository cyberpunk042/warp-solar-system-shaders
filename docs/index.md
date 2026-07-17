# Warp Shaders

**A hyper-realistic procedural rendering engine for [NVIDIA Warp](https://github.com/NVIDIA/warp).**

Per-pixel `@wp.kernel` shaders — SDF raymarching, procedural noise, PBR,
physically based atmosphere, and volumetrics — written in Python, JIT-compiled
to **CUDA when a GPU is present and to CPU otherwise**, and driven by **one
quality knob** so the same scene renders on a laptop CPU and scales to a
high-end GPU.

<p align="center">
  <img src="engine/earth_v2.png" width="32%" alt="Earth v2">
  <img src="engine/gas_giant.png" width="32%" alt="Gas giant">
  <img src="engine/galaxy.png" width="32%" alt="Spiral galaxy">
</p>

```python
import warp as wp
import warp_shaders as ws

wp.init()
ws.set_active("high")                       # pick a quality tier
img = ws.render("earth_v2", width=1280, height=720, time=0.0)   # (H, W, 3) float
```

## Why an engine, not a demo

Every technique here is a **reusable building block with a cited primary
source**, composable into your own kernels:

| Layer | Module | What you get |
|---|---|---|
| **Procedural** | [`ws.procedural`](api/procedural.md) | value / Perlin / simplex / Worley / fBm / ridged / billow / domain-warp / curl noise **with analytic derivatives** and **seamless tiling**, plus a full SDF primitive + boolean-operator library |
| **Shading** | [`ws.engine`](api/engine.md) | `@wp.struct` uniforms (camera / light / frame / quality), GGX Cook-Torrance **PBR**, a typed **Material**, physically based **atmosphere** (+ transmittance LUT), **volumetric** clouds, and a host **post** pipeline (ACES/AgX, bloom, godrays, vignette) |
| **Textures** | [`ws.textures`](api/textures.md) | portable **2D / 3D / equirectangular** sampling over `wp.array` — image maps, baked LUTs, and 3D noise volumes with no hardware-texture dependency |
| **Scale** | [`ws.lod`](api/lod.md) | four **quality tiers** (`low` / `medium` / `high` / `ultra`) that scale every sample-count/octave cost from one knob; auto-detected per device |
| **Scenes** | [`ws.scene`](api/scene.md) | a zero-config **registry** — drop a module in `warp_shaders/scenes/`, it's discovered automatically |

## Where to go next

- **[Quickstart](quickstart.md)** — install, render a scene, and write your first 30-line shader.
- **[Concepts](concepts.md)** — the architecture: device vs host, the render pipeline, quality tiers, portability, and Warp gotchas.
- **[Writing a scene](guides/writing-a-scene.md)** — the full guide, from `map()` to post.
- **[API reference](api/index.md)** — every public symbol, grouped by subsystem.
- **[Gallery](gallery.md)** — all 308 scenes.

## Install

```bash
pip install warp-lang numpy pillow      # runtime deps
git clone https://github.com/cyberpunk042/warp-solar-system-shaders
cd warp-solar-system-shaders
python render.py --list                 # see every scene
python render.py --scene earth_v2 --quality high -o earth.png
```

No GPU required — Warp falls back to CPU automatically. A CUDA device is used
when present.
