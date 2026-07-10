# warp-solar-system-shaders

Raymarched GPU scenes written with [**NVIDIA Warp**](https://github.com/NVIDIA/warp)
(`warp-lang`). Shader-style per-pixel kernels — SDF raymarching, value noise,
fBm — written in Python `@wp.kernel` / `@wp.func` form, running on CUDA when a
GPU is present and transparently falling back to CPU otherwise.

The flagship scene is a solar system: a procedurally-textured planet with a
plasma jet, magnetic field rings, orbiting probes, and a cube-mapped starfield —
a Warp port of the GLSL Shadertoy original kept at
[`reference/solar-system.frag`](reference/solar-system.frag).

![preview](docs/preview.png)

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`warp-lang` ships its own CPU/CUDA codegen — no separate CUDA toolkit needed for
CPU rendering. On a machine with an NVIDIA GPU + driver, Warp uses it
automatically.

## Render

```bash
# single frame (auto device: CUDA if available, else CPU)
python render.py --time 2.0 --width 1280 --height 720 -o frame.png

# a spinning GIF
python render.py --frames 60 --fps 30 --gif out/spin.gif

# PNG frame sequence
python render.py --frames 120 --out-dir out/frames

# force CPU (works anywhere; slower)
python render.py --device cpu --width 640 --height 360 -o frame.png
```

`--mouse MX MY` orbits the camera, matching the shader's `iMouse` convention
(pixel coordinates).

## Write your own scene

`warp_shaders/sdf.py` is a reusable toolkit of `@wp.func` building blocks —
`hash2d`, `noise2d`, `fbm2d`, `rot2`, `sd_torus`, `fract`. Compose them inside a
`@wp.kernel` that writes a `wp.vec3` per pixel:

```python
import warp as wp
from warp_shaders.sdf import fbm2d, rot2

@wp.kernel
def my_scene(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float):
    i, j = wp.tid()                        # i = row, j = column
    uv = wp.vec2((float(j) + 0.5) / float(width),
                 (float(i) + 0.5) / float(height))
    img[i, j] = wp.vec3(uv[0], uv[1], 0.5 + 0.5 * wp.sin(time))

img = wp.zeros((h, w), dtype=wp.vec3, device="cpu")
wp.launch(my_scene, dim=(h, w), inputs=[img, w, h, t], device="cpu")
frame = img.numpy()                        # (H, W, 3) float array
```

### GLSL → Warp cheatsheet

Porting a Shadertoy shader is mostly mechanical. The main friction is that Warp
has no swizzles and distinguishes scalars from vectors:

| GLSL | Warp |
|---|---|
| `mix(a, b, t)` | `wp.lerp(a, b, t)` |
| `fract(x)` | `x - wp.floor(x)` (or `sdf.fract`) |
| `atan(y, x)` | `wp.atan2(y, x)` |
| `p.xz = rotate(p.xz, a)` | rebuild: `r = rot2(wp.vec2(p[0], p[2]), a); p = wp.vec3(r[0], p[1], r[1])` |
| `v.x` / `v.y` / `v.z` | `v[0]` / `v[1]` / `v[2]` |
| `void f(out float m)` | return a tuple: `return dist, m` |
| `iResolution`, `iTime`, `iMouse` | kernel arguments you pass at launch |

See `warp_shaders/solar_system.py` next to `reference/solar-system.frag` for a
full worked example.

## Layout

```
render.py                     CLI: single frame, PNG sequence, or GIF
warp_shaders/
  sdf.py                      reusable @wp.func toolkit (hash/noise/fbm/rot/SDF)
  solar_system.py             the scene kernel + render() helper
reference/
  solar-system.frag           original GLSL shader (provenance / cross-check)
docs/preview.png              rendered still
requirements.txt
```

## Why Warp instead of a GLSL player

Warp kernels are ordinary Python that JIT-compile to native CUDA/CPU, so the
same raymarcher is scriptable, differentiable-capable, and composes with NumPy
and the rest of a simulation pipeline — while still reading like a shader.
