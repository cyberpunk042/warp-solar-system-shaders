# `warp_shaders.scene`

The scene contract and the auto-discovery registry. All **host**.

```python
import warp_shaders as ws
ws.list_scenes()                 # every registered Scene
ws.render("earth_v2", width=1280, height=720, time=0.0)
```

## `Scene`

A renderable scene. Construct one at module level in `warp_shaders/scenes/` and
the registry finds it — no central list to edit.

| Field | Type | Meaning |
|---|---|---|
| `name` | `str` | unique registry key (also the `--scene` value) |
| `description` | `str` | one-line summary (shown by `--list`) |
| `kernel` | `wp.Kernel` | a `@wp.kernel` with the fixed signature `(img, width, height, time, mouse)` |
| `renderer` | callable | `renderer(width, height, time, mouse, device) -> (H, W, 3)` array |
| `width` / `height` | `int` | default resolution (`960 × 540`) |

Set **either** `kernel` **or** `renderer`:

- **`kernel`** — a self-contained shader with the 5-argument contract. The
  registry launches it for you (`img` is a `wp.array2d(vec3)`, `mouse` a
  `wp.vec2`).
- **`renderer`** — a callback you control. Use this when you need uniform
  structs, the LOD tier, baked textures/LUTs, or host post — it returns the
  finished `(H, W, 3)` array itself.

### `Scene.render`

`render(width=None, height=None, time=0.0, mouse=(0,0), device="cpu") -> ndarray`
— render one frame; returns an `(H, W, 3)` float array (unclamped HDR for kernel
scenes, or whatever the renderer returns).

## Registry functions

| Function | Signature | Notes |
|---|---|---|
| `list_scenes` | `() -> list[Scene]` | all scenes, sorted by name |
| `get_scene` | `(name: str) -> Scene` | look up by name; raises with the available names if unknown |
| `render` | `(scene_name: str, **kwargs) -> ndarray` | convenience — `get_scene(name).render(**kwargs)` |

## Discovery rules

`list_scenes()` imports every non-underscore module in `warp_shaders/scenes/`
and collects:

- a module-level `SCENE` (a single `Scene`), and/or
- a module-level `SCENES` (a list of `Scene`, e.g. one per chemical element).

See [Writing a scene](../guides/writing-a-scene.md) for the two patterns end to
end.
