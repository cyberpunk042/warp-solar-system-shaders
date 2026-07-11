"""Render engine — the shading half of Warp Shaders.

Everything a scene needs to turn a ray into a pixel:

- **uniforms** — `@wp.struct` blocks (:class:`Camera`, :class:`Light`,
  :class:`Frame`, :class:`Quality`) + the primary ray generator
  :func:`camera_ray_dir`, plus host builders (``make_camera``, ...).
- **material** — a typed surface (:class:`Material`) and its shader
  :func:`shade_material`.
- **pbr** — GGX Cook-Torrance building blocks (:func:`shade_pbr`,
  :func:`fresnel_schlick`, :func:`distribution_ggx`, ...).
- **atmosphere** — physically based sky: single-scatter
  (:func:`atmosphere`, :func:`sky_radiance`) + precomputed transmittance and
  Hillaire multiple-scattering LUTs (``build_transmittance_lut`` /
  ``build_multiscatter_lut``, :func:`atmosphere_lut`).
- **volumetric** — cloud density + light-marching
  (:func:`cloud_density`, :func:`march_clouds`, :func:`hg_phase`) over a baked
  seamless detail volume (``build_cloud_detail``).
- **post** — host-side tonemap / bloom / godrays / vignette.
- **shading** — small map-independent helpers (:func:`apply_fog`,
  :func:`sun_disk`, :func:`sky_gradient`).

Device functions (``@wp.func``) are meant to be *called inside your own
``@wp.kernel``*; host helpers (``make_*``, ``post.*``, ``build_*_lut``) run in
plain Python.
"""

from . import atmosphere, color, imageio, intersect, pbr, post, shadow, video, volumetric
from .imageio import RenderTarget, load_hdr, save_hdr, save_npy, save_png
from .video import write_video
from .color import (blackbody, kelvin_to_rgb, linear_to_srgb, luminance,
                    srgb_to_linear)
from .intersect import (ray_box, ray_disk, ray_plane, ray_sphere, ray_sphere_o,
                        sphere_t)
from .material import Material, make_mat, make_material, shade_material
from .sky import milky_way, starfield
from .pbr import (
    distribution_ggx, fresnel_schlick, geometry_schlick_ggx, geometry_smith,
    shade_pbr,
)
from .shading import apply_fog, sky_gradient, sun_disk
from .shadow import (
    ground_contact_ao, penumbra, soft_shadow_sphere, sphere_ao, sphere_occlusion,
)
from .uniforms import (
    Camera, Frame, Light, Quality, camera_ray_dir, focus_point, lens_offset,
    make_camera, make_frame, make_light, make_quality,
)

__all__ = [
    # subsystem namespaces
    "atmosphere", "color", "imageio", "intersect", "pbr", "post", "shadow",
    "video", "volumetric",
    # frame output (host)
    "RenderTarget", "save_png", "save_hdr", "save_npy", "load_hdr", "write_video",
    # colour science (device)
    "kelvin_to_rgb", "blackbody", "luminance", "linear_to_srgb", "srgb_to_linear",
    # ray-primitive intersection (device)
    "ray_sphere", "ray_sphere_o", "sphere_t", "ray_plane", "ray_disk", "ray_box",
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
    # analytic soft shadows + ambient occlusion (device)
    "soft_shadow_sphere", "sphere_occlusion", "sphere_ao", "penumbra",
    "ground_contact_ao",
    # sky backgrounds (device)
    "starfield", "milky_way",
]
