"""Render engine — the shading half of Warp Shaders.

Everything a scene needs to turn a ray into a pixel:

- **uniforms** — `@wp.struct` blocks (:class:`Camera`, :class:`Light`,
  :class:`Frame`, :class:`Quality`) + the primary ray generator
  :func:`camera_ray_dir`, plus host builders (``make_camera``, ...).
- **material** — a typed surface (:class:`Material`) and its shader
  :func:`shade_material`.
- **pbr** — GGX Cook-Torrance building blocks (:func:`shade_pbr`,
  :func:`fresnel_schlick`, :func:`distribution_ggx`, ...).
- **atmosphere** — physically based single-scatter sky
  (:func:`atmosphere`, :func:`sky_radiance`) + a precomputed transmittance
  LUT (``build_transmittance_lut``, :func:`atmosphere_lut`).
- **volumetric** — cloud density + light-marching
  (:func:`cloud_density`, :func:`march_clouds`, :func:`hg_phase`).
- **post** — host-side tonemap / bloom / godrays / vignette.
- **shading** — small map-independent helpers (:func:`apply_fog`,
  :func:`sun_disk`, :func:`sky_gradient`).

Device functions (``@wp.func``) are meant to be *called inside your own
``@wp.kernel``*; host helpers (``make_*``, ``post.*``, ``build_*_lut``) run in
plain Python.
"""

from . import atmosphere, pbr, post, volumetric
from .material import Material, make_mat, make_material, shade_material
from .pbr import (
    distribution_ggx, fresnel_schlick, geometry_schlick_ggx, geometry_smith,
    shade_pbr,
)
from .shading import apply_fog, sky_gradient, sun_disk
from .uniforms import (
    Camera, Frame, Light, Quality, camera_ray_dir, focus_point, lens_offset,
    make_camera, make_frame, make_light, make_quality,
)

__all__ = [
    # subsystem namespaces
    "atmosphere", "pbr", "post", "volumetric",
    # uniforms
    "Camera", "Light", "Frame", "Quality", "camera_ray_dir",
    "lens_offset", "focus_point",
    "make_camera", "make_light", "make_frame", "make_quality",
    # material
    "Material", "make_mat", "make_material", "shade_material",
    # pbr device functions
    "shade_pbr", "fresnel_schlick", "distribution_ggx",
    "geometry_schlick_ggx", "geometry_smith",
    # reusable shading helpers
    "apply_fog", "sun_disk", "sky_gradient",
]
