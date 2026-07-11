"""Material system — a typed surface description for the PBR shader.

Bundles the per-surface parameters (albedo, roughness, metallic, emission) into a
`@wp.struct` so scenes describe materials once and shade them uniformly, instead of
threading loose scalars. `shade_material` wraps `engine.pbr.shade_pbr` and adds
self-emission.
"""

import warp as wp

from .pbr import shade_pbr


@wp.struct
class Material:
    albedo: wp.vec3
    roughness: float
    metallic: float
    emission: wp.vec3


@wp.func
def shade_material(m: Material, n: wp.vec3, v: wp.vec3, l: wp.vec3,
                   light_color: wp.vec3, light_intensity: float) -> wp.vec3:
    """Direct PBR radiance from one light, plus the material's self-emission."""
    direct = shade_pbr(n, v, l, m.albedo, m.roughness, m.metallic, light_color)
    return direct * light_intensity + m.emission


@wp.func
def make_mat(albedo: wp.vec3, roughness: float, metallic: float) -> Material:
    """Construct a (non-emissive) material inside a kernel."""
    m = Material()
    m.albedo = albedo
    m.roughness = roughness
    m.metallic = metallic
    m.emission = wp.vec3(0.0, 0.0, 0.0)
    return m


def make_material(albedo, roughness=0.5, metallic=0.0, emission=(0.0, 0.0, 0.0)) -> Material:
    """Host-side material builder."""
    m = Material()
    m.albedo = wp.vec3(*[float(x) for x in albedo])
    m.roughness = float(roughness)
    m.metallic = float(metallic)
    m.emission = wp.vec3(*[float(x) for x in emission])
    return m
