"""warp_shaders — raymarched GPU scenes written with NVIDIA Warp.

Scenes live in :mod:`warp_shaders.scenes` (one module each) and are discovered
through the registry in :mod:`warp_shaders.scene`.
"""

from .scene import Scene, get_scene, list_scenes, render

__all__ = ["Scene", "get_scene", "list_scenes", "render"]
