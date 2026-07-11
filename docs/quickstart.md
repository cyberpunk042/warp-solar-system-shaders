# Quickstart

## Install

```bash
pip install warp-lang numpy pillow
git clone https://github.com/cyberpunk042/warp-solar-system-shaders
cd warp-solar-system-shaders
```

Warp compiles to CUDA when a GPU is present and to CPU otherwise — nothing to
configure.

## Render a built-in scene

From the command line:

```bash
python render.py --list                              # every scene name
python render.py --scene pbr_demo --quality high -o out.png
python render.py --scene sky --quality ultra --ss 2  # 2x supersampled
python render.py --scene earth_v2 --gif spin.gif --frames 60 --fps 30
```

Or from Python:

```python
import warp as wp
import warp_shaders as ws

wp.init()
ws.set_active("high")                    # low | medium | high | ultra | auto
img = ws.render("pbr_demo", width=960, height=540, time=0.0)
# img is a (540, 960, 3) float array in [0, 1], ready to save
```

The CLI flags that matter:

| Flag | Meaning |
|---|---|
| `--scene NAME` | which scene (see `--list`) |
| `--quality {auto,low,medium,high,ultra}` | LOD tier; `auto` picks by device |
| `--ss N` | supersample N× then downscale (anti-aliasing) |
| `--width` / `--height` | output resolution |
| `--time T` | scene time in seconds (for a single frame) |
| `--frames` / `--fps` / `--gif` | render an animation |
| `--device {auto,cpu,cuda}` | force a device |

## Write your first scene

A scene is one file in `warp_shaders/scenes/`. The registry finds it by the
module-level `SCENE` object — no central list to edit. Here is a complete,
LOD-aware shader that raymarches a single procedural-noise-displaced sphere with
PBR shading and a post pipeline:

```python
# warp_shaders/scenes/my_blob.py
import warp as wp

from ..engine import post
from ..engine.material import make_mat, shade_material
from ..engine.uniforms import (
    Camera, Frame, Light, Quality, camera_ray_dir,
    make_camera, make_frame, make_light, make_quality,
)
from ..lod import active_tier
from ..procedural.noise import fbm3
from ..procedural.sdf import sd_sphere
from ..scene import Scene


@wp.func
def _map(p: wp.vec3, time: float) -> float:
    # a sphere whose surface is pushed around by animated fBm
    disp = 0.15 * fbm3(p * 2.0 + wp.vec3(0.0, 0.0, time * 0.2), 5)
    return sd_sphere(p, 1.0) + disp


@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.001
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e, 0.0, 0.0), time) - _map(p - wp.vec3(e, 0.0, 0.0), time),
        _map(p + wp.vec3(0.0, e, 0.0), time) - _map(p - wp.vec3(0.0, e, 0.0), time),
        _map(p + wp.vec3(0.0, 0.0, e), time) - _map(p - wp.vec3(0.0, 0.0, e), time),
    ))


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, light: Light,
                  qual: Quality, frame: Frame):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(frame.width)) - 1.0
    v = (2.0 * (float(frame.height - 1 - i) + 0.5) / float(frame.height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.0)
    hit = int(0)
    for _ in range(qual.raymarch_steps):          # tier-scaled step budget
        p = ro + rd * t
        d = _map(p, frame.time)
        if d < 0.001:
            hit = 1
            break
        t += d * 0.9
        if t > 8.0:
            break

    if hit == 0:
        img[i, j] = wp.vec3(0.02, 0.03, 0.05)     # background
    else:
        p = ro + rd * t
        n = _normal(p, frame.time)
        mat = make_mat(wp.vec3(0.8, 0.3, 0.2), 0.35, 0.0)
        img[i, j] = shade_material(mat, n, -rd, light.dir,
                                   light.color, light.intensity)


def _render(width, height, time, mouse, device):
    tier = active_tier()
    cam = make_camera((0.0, 0.0, 3.5), (0.0, 0.0, 0.0),
                      fov_deg=45.0, aspect=width / height)
    light = make_light((0.6, 0.7, 0.4), (1.0, 0.95, 0.9), 3.0)
    qual = make_quality(tier)
    frame = make_frame(time, width, height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, light, qual, frame], device=device)
    wp.synchronize_device(device)

    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(name="my_blob",
              description="fBm-displaced sphere with PBR. --quality low..ultra.",
              renderer=_render)
```

Render it:

```bash
python render.py --scene my_blob --quality medium -o blob.png
```

That's the whole loop. The next stops:

- **[Concepts](concepts.md)** — why the kernel/host split, how tiers flow into
  kernels, and what portability buys you.
- **[Writing a scene](guides/writing-a-scene.md)** — normals, soft shadows,
  ambient occlusion, materials, atmosphere, and the two `Scene` shapes.
