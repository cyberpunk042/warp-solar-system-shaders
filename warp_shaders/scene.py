"""Scene framework — the uniform contract every Warp scene implements.

Adding a shader is one file. Drop ``warp_shaders/scenes/<name>.py`` that defines
a kernel with the standard signature and exposes a module-level ``SCENE``::

    import warp as wp
    from ..scene import Scene
    from ..sdf import fbm2d, rot2   # reusable toolkit

    @wp.kernel
    def render_kernel(img: wp.array2d(dtype=wp.vec3),
                      width: int, height: int, time: float, mouse: wp.vec2):
        i, j = wp.tid()
        ...
        img[i, j] = wp.vec3(r, g, b)

    SCENE = Scene(name="my_scene", kernel=render_kernel, description="...")

The registry auto-discovers every non-underscore module in ``scenes/``. No
central list to edit — that's what "ready for many" means.

The kernel signature is fixed so the launcher is uniform:
``(img: array2d(vec3), width: int, height: int, time: float, mouse: vec2)``.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil

import numpy as np
import warp as wp


@dataclasses.dataclass
class Scene:
    """A renderable Warp scene."""

    name: str
    kernel: object = None  # a wp.Kernel (the @wp.kernel-decorated function)
    description: str = ""
    width: int = 960
    height: int = 540
    # Optional custom renderer(width, height, time, mouse, device) -> (H,W,3) array.
    # Scenes that need extra per-scene inputs (arrays, parameters) set this instead
    # of relying on the fixed 5-argument kernel contract.
    renderer: object = None

    def render(self, width: int | None = None, height: int | None = None,
               time: float = 0.0, mouse=(0.0, 0.0), device: str = "cpu") -> np.ndarray:
        """Render one frame. Returns an ``(H, W, 3)`` float32 array (unclamped)."""
        w = width or self.width
        h = height or self.height
        if self.renderer is not None:
            return self.renderer(w, h, float(time), mouse, device)
        img = wp.zeros((h, w), dtype=wp.vec3, device=device)
        wp.launch(
            self.kernel,
            dim=(h, w),
            inputs=[img, w, h, float(time), wp.vec2(float(mouse[0]), float(mouse[1]))],
            device=device,
        )
        wp.synchronize_device(device)
        return img.numpy()


_REGISTRY: dict[str, Scene] | None = None


def _discover() -> dict[str, Scene]:
    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY

    from . import scenes as scenes_pkg

    registry: dict[str, Scene] = {}
    for info in pkgutil.iter_modules(scenes_pkg.__path__):
        if info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{scenes_pkg.__name__}.{info.name}")
        scene = getattr(module, "SCENE", None)
        if isinstance(scene, Scene):
            registry[scene.name] = scene
        # A module may also expose SCENES (a list) — e.g. one per chemical element.
        for s in getattr(module, "SCENES", []) or []:
            if isinstance(s, Scene):
                registry[s.name] = s
    _REGISTRY = registry
    return registry


def list_scenes() -> list[Scene]:
    """All registered scenes, sorted by name."""
    return sorted(_discover().values(), key=lambda s: s.name)


def get_scene(name: str) -> Scene:
    """Look up a scene by name, or raise with the available names."""
    registry = _discover()
    if name not in registry:
        available = ", ".join(sorted(registry)) or "(none)"
        raise KeyError(f"unknown scene '{name}'. Available: {available}")
    return registry[name]


def render(scene_name: str, **kwargs) -> np.ndarray:
    """Convenience: render a scene by name. Kwargs forwarded to ``Scene.render``."""
    return get_scene(scene_name).render(**kwargs)
