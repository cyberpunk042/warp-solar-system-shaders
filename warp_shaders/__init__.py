"""Warp Shaders — a hyper-realistic procedural rendering engine for NVIDIA Warp.

Everything below runs as per-pixel `@wp.kernel` code JIT-compiled by
`warp-lang <https://github.com/NVIDIA/warp>`_ to CUDA when a GPU is present and
to CPU otherwise, driven by **one quality knob** so the same scene renders on a
laptop CPU and scales to a high-end GPU.

The engine is organised in layers, each importable on its own::

    import warp_shaders as ws

    ws.procedural   # noise (value/Perlin/simplex/Worley/fbm/...) + SDF library
    ws.engine       # uniforms, PBR, materials, atmosphere, volumetrics, post
    ws.textures     # portable 2D/3D/equirect sampling over wp.array
    ws.lod          # quality tiers (low / medium / high / ultra)
    ws.scene        # the scene registry

Quickstart — render a built-in scene to an HDR array::

    import warp as wp, warp_shaders as ws
    wp.init()
    ws.set_active("high")
    img = ws.render("pbr_demo", width=960, height=540, time=0.0)

Quickstart — write your own scene by calling engine device functions from a
``@wp.kernel``; see :mod:`warp_shaders.engine` and
:mod:`warp_shaders.procedural`, or ``docs/guides/writing-a-scene.md``.

**Device vs host.** Symbols marked ``@wp.func`` are *device functions* — call
them inside your own ``@wp.kernel``. Symbols named ``make_*`` / ``post.*`` /
``build_*_lut`` / ``set_active`` run in ordinary Python (host side).
"""

__version__ = "0.1.0"

# --- subsystem namespaces (device functions live here; import inside kernels) ---
from . import engine, life, lod, procedural, textures

# --- scene registry (host) ---
from .scene import Scene, get_scene, list_scenes, render

# --- quality tiers (host) ---
from .lod import (
    QualityTier, active_tier, auto_tier, get_tier, set_active,
)

# --- engine uniforms + material + post (curated re-exports) ---
from .engine import post
from .engine.material import Material, make_material
from .engine.uniforms import (
    Camera, Frame, Light, Quality, make_camera, make_frame, make_light,
    make_quality,
)

__all__ = [
    "__version__",
    # namespaces
    "engine", "procedural", "textures", "lod", "post", "life",
    # scenes
    "Scene", "get_scene", "list_scenes", "render",
    # quality tiers
    "QualityTier", "get_tier", "auto_tier", "set_active", "active_tier",
    # uniforms + material
    "Camera", "Light", "Frame", "Quality",
    "make_camera", "make_light", "make_frame", "make_quality",
    "Material", "make_material",
]
