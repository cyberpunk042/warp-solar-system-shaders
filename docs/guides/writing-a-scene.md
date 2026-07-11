# Writing a scene

A scene is one self-contained module in `warp_shaders/scenes/`. The registry
discovers it by a module-level `SCENE` object, so **adding a scene is adding a
file** — there is no central list to edit. This guide covers the full anatomy;
for the minimal version see the [Quickstart](../quickstart.md#write-your-first-scene).

## The two shapes

A [`Scene`](../api/scene.md) is built with **either** a `kernel` **or** a
`renderer`:

### 1. Kernel scene

A self-contained shader with the fixed 5-argument contract
`(img, width, height, time, mouse)`. The registry launches it for you. Good for
small shaders that don't need uniforms or host post.

```python
import warp as wp
from ..scene import Scene

@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int,
                  time: float, mouse: wp.vec2):
    i, j = wp.tid()
    u = float(j) / float(width)
    v = float(i) / float(height)
    img[i, j] = wp.vec3(u, v, 0.5 + 0.5 * wp.sin(time))

SCENE = Scene(name="uv", description="UV gradient.", kernel=render_kernel)
```

### 2. Renderer scene

A `renderer(width, height, time, mouse, device)` callback that returns the
finished `(H, W, 3)` array. Use this whenever you need `@wp.struct` uniforms,
the LOD tier, baked textures/LUTs, or the host post pipeline — i.e. almost every
realistic scene. This is the shape the rest of the guide uses.

## Anatomy of a realistic renderer scene

Most hero scenes are variations on one skeleton. `scenes/pbr_demo.py` is the
canonical copy-me template.

### The scene `map()` (inlined)

Warp has **no function pointers**, so a raymarcher can't take a `map` callback —
you *inline* the scene distance field as a module-level `@wp.func`:

```python
from ..procedural.sdf import sd_sphere, op_smooth_union

@wp.func
def _map(p: wp.vec3, time: float) -> float:
    plane = p[1] + 1.0
    ball  = sd_sphere(p - wp.vec3(0.0, 0.15 * wp.sin(time), 0.0), 0.7)
    return op_smooth_union(plane, ball, 0.25)
```

### Normals, shadows, AO

Standard SDF techniques, all built from `_map`:

```python
@wp.func
def _normal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0015
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e,0,0), time) - _map(p - wp.vec3(e,0,0), time),
        _map(p + wp.vec3(0,e,0), time) - _map(p - wp.vec3(0,e,0), time),
        _map(p + wp.vec3(0,0,e), time) - _map(p - wp.vec3(0,0,e), time)))

@wp.func
def _soft_shadow(ro: wp.vec3, rd: wp.vec3, time: float, steps: int) -> float:
    res = float(1.0); t = float(0.04)
    for _ in range(steps):                       # tier-scaled
        h = _map(ro + rd * t, time)
        if h < 0.001: return 0.0
        res = wp.min(res, 10.0 * h / t)
        t += wp.clamp(h, 0.02, 0.3)
    return wp.clamp(res, 0.0, 1.0)
```

For a noise-displaced surface, prefer the **analytic gradient** from
`noised3` (see [procedural](../api/procedural.md#noise)) over finite differences.

### The render kernel

Start every kernel with `camera_ray_dir`, sphere-trace with the tier's step
budget, then shade:

```python
from ..engine.uniforms import Camera, Frame, Light, Quality, camera_ray_dir
from ..engine.material import make_mat, shade_material

@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, light: Light,
                  qual: Quality, frame: Frame):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(frame.width)) - 1.0
    v = (2.0 * (float(frame.height - 1 - i) + 0.5) / float(frame.height)) - 1.0
    ro, rd = cam.eye, camera_ray_dir(cam, u, v)

    t = float(0.0); hit = int(0)
    for _ in range(qual.raymarch_steps):
        d = _map(ro + rd * t, frame.time)
        if d < 0.0008 * t + 0.0004:
            hit = 1; break
        t += d * 0.9
        if t > 30.0: break

    if hit == 0:
        img[i, j] = wp.vec3(0.02, 0.03, 0.05)
    else:
        p = ro + rd * t
        n = _normal(p, frame.time)
        sh = _soft_shadow(p + n * 0.01, light.dir, frame.time, qual.shadow_steps)
        mat = make_mat(wp.vec3(0.8, 0.3, 0.2), 0.35, 0.0)
        img[i, j] = shade_material(mat, n, -rd, light.dir,
                                   light.color, light.intensity * sh)
```

### The host renderer

Read the tier, pack the uniforms, launch, then run post:

```python
from ..engine import post
from ..engine.uniforms import make_camera, make_light, make_quality, make_frame
from ..lod import active_tier

def _render(width, height, time, mouse, device):
    tier  = active_tier()
    cam   = make_camera((0,0,5), (0,0,0), fov_deg=42.0, aspect=width/height)
    light = make_light((0.5, 0.75, 0.4), (1.0, 0.96, 0.9), 3.2)
    qual  = make_quality(tier)
    frame = make_frame(time, width, height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, light, qual, frame], device=device)
    wp.synchronize_device(device)

    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=6, passes=2)
    out = post.tonemap(hdr, mode="aces", exposure=1.15)
    return post.vignette(out, 0.25)

SCENE = Scene(name="my_scene", description="... --quality low..ultra.",
              renderer=_render)
```

## Make it LOD-aware

The whole point of the tier system: **read `active_tier()` and thread its counts
into the kernel.** `make_quality(tier)` packs `raymarch_steps`, `shadow_steps`,
`ao_steps`, `noise_octaves`, `volumetric_steps` into the `Quality` uniform;
then `for _ in range(qual.raymarch_steps)` scales for free. For atmosphere/LUT
work, size the LUT with `tier.lut_size` and use `sample_counts(tier.name)`.
Verify your scene across `--quality low medium high ultra`.

## Warp gotchas

These bite everyone once (full list in [Concepts](../concepts.md#warp-gotchas-worth-knowing-up-front)):

- **No `vec * vec`** — component-wise multiply is `wp.cw_mul(a, b)`. `vec * scalar` is fine.
- **No function pointers** — inline your `map()`; reuse the map-independent
  funcs (PBR, atmosphere, post, noise, SDF).
- **No `import` inside `@wp.func`** — imports at module top level.
- **Dynamic loops OK** — `for _ in range(runtime_int)` compiles; that's how tier
  counts flow in.
- **Verify with `render.py`**, not `python -c` — Warp can't compile a kernel
  from an `exec`'d string.

## Checklist before you commit

- [ ] `python render.py --scene NAME --quality low` renders without errors
- [ ] Renders across `--quality low medium high ultra`
- [ ] `python -m tests.test_public_api` still passes (registry stays healthy)
- [ ] A preview PNG in `docs/engine/` and a row in the [gallery](../gallery.md)
