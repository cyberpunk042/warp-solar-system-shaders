# warp-solar-system-shaders

Raymarched GPU scenes written with [**NVIDIA Warp**](https://github.com/NVIDIA/warp)
(`warp-lang`). Shader-style per-pixel kernels — SDF raymarching, value noise,
fBm — written in Python `@wp.kernel` / `@wp.func` form, running on CUDA when a
GPU is present and transparently falling back to CPU otherwise.

It's a **multi-scene gallery**: each shader is one self-contained module in
`warp_shaders/scenes/`, auto-discovered by a registry. Adding a scene is adding
a file — no central list to edit.

The flagship scene is a **neutron star**: a dense pulsar core with relativistic
jets along the magnetic axis, magnetic field rings, orbiting matter, and a
cube-mapped starfield — a Warp port of the GLSL Shadertoy original kept at
[`reference/neutron-star.frag`](reference/neutron-star.frag).

![neutron star](docs/preview.png)

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
python render.py --list                         # show available scenes

# single frame (auto device: CUDA if available, else CPU)
python render.py --scene neutron_star --time 2.0 --width 1280 --height 720 -o frame.png

# a spinning GIF
python render.py --scene neutron_star --frames 60 --fps 30 --gif out/spin.gif

# PNG frame sequence
python render.py --scene neutron_star --frames 120 --out-dir out/frames

# force CPU (works anywhere; slower)
python render.py --scene neutron_star --device cpu --width 640 --height 360 -o frame.png
```

`--mouse MX MY` drives the camera/pan, matching the shader's `iMouse` convention
(pixel coordinates).

## Add a scene (the workflow for every new shader)

1. Copy the template:

   ```bash
   cp warp_shaders/scenes/_template.py warp_shaders/scenes/my_scene.py
   ```

2. Write the kernel and set the `SCENE` name. Every scene implements the **same
   kernel contract**, which is what keeps the launcher uniform:

   ```python
   @wp.kernel
   def render_kernel(img: wp.array2d(dtype=wp.vec3),
                     width: int, height: int, time: float, mouse: wp.vec2):
       i, j = wp.tid()          # i = row, j = column
       ...
       img[i, j] = wp.vec3(r, g, b)

   SCENE = Scene(name="my_scene", kernel=render_kernel, description="...")
   ```

3. It's live immediately:

   ```bash
   python render.py --list
   python render.py --scene my_scene -o my_scene.png
   ```

Underscore-prefixed modules (like `_template.py`) are skipped by discovery.

### GLSL → Warp cheatsheet

Porting a Shadertoy shader is mostly mechanical. The main friction is that Warp
has no swizzles and distinguishes scalars from vectors:

| GLSL | Warp |
|---|---|
| `mainImage(out vec4 c, in vec2 fragCoord)` | the `render_kernel` body |
| `iResolution` / `iTime` / `iMouse` | `width, height` / `time` / `mouse` kernel args |
| `mix(a, b, t)` | `wp.lerp(a, b, t)` |
| `fract(x)` | `x - wp.floor(x)` (or `sdf.fract`) |
| `atan(y, x)` | `wp.atan2(y, x)` |
| `p.xz = rotate(p.xz, a)` | rebuild: `r = rot2(wp.vec2(p[0], p[2]), a); p = wp.vec3(r[0], p[1], r[1])` |
| `v.x` / `v.y` / `v.z` | `v[0]` / `v[1]` / `v[2]` |
| `void f(out float m)` | return a tuple: `return dist, m` |

Reusable building blocks live in `warp_shaders/sdf.py` (`hash2d`, `noise2d`,
`fbm2d`, `rot2`, `sd_torus`, `fract`). Grow that toolkit as scenes share more
primitives. See `warp_shaders/scenes/neutron_star.py` next to
`reference/neutron-star.frag` for a full worked port.

## Layout

```
render.py                        CLI: --list, --scene, single frame / sequence / GIF
warp_shaders/
  scene.py                       Scene contract + auto-discovery registry
  sdf.py                         reusable @wp.func toolkit (hash/noise/fbm/rot/SDF)
  scenes/
    neutron_star.py              flagship pulsar scene
    starfield.py                 minimal second scene (registry demo)
    _template.py                 copy-me starter (skipped by discovery)
reference/
  neutron-star.frag              original GLSL shader (provenance / cross-check)
docs/preview.png                 rendered still
requirements.txt
```

## Why Warp instead of a GLSL player

Warp kernels are ordinary Python that JIT-compile to native CUDA/CPU, so the
same raymarcher is scriptable, differentiable-capable, and composes with NumPy
and the rest of a simulation pipeline — while still reading like a shader.
