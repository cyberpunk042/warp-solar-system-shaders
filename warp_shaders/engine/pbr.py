"""Physically based shading — Cook-Torrance microfacet BRDF (Warp device funcs).

GGX/Trowbridge-Reitz normal distribution + Smith geometry + Schlick Fresnel, with
energy-conserving Lambert diffuse. Maps directly onto raymarched SDF hits and
analytic surfaces (e.g. the Earth ocean).

Sources (docs/research): Cook & Torrance 1982; Walter et al. 2007 (GGX);
Schlick 1994 (Fresnel); Karis 2013 (UE4 PBR notes, k = (r+1)^2/8).
"""

import warp as wp

_PI = 3.14159265


@wp.func
def fresnel_schlick(cos_theta: float, f0: wp.vec3) -> wp.vec3:
    m = wp.clamp(1.0 - cos_theta, 0.0, 1.0)
    m2 = m * m
    return f0 + (wp.vec3(1.0, 1.0, 1.0) - f0) * (m2 * m2 * m)


@wp.func
def distribution_ggx(n_dot_h: float, rough: float) -> float:
    a = rough * rough
    a2 = a * a
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    return a2 / (_PI * d * d + 1e-7)


@wp.func
def geometry_schlick_ggx(n_dot_x: float, rough: float) -> float:
    r = rough + 1.0
    k = (r * r) / 8.0
    return n_dot_x / (n_dot_x * (1.0 - k) + k)


@wp.func
def geometry_smith(n_dot_v: float, n_dot_l: float, rough: float) -> float:
    return geometry_schlick_ggx(n_dot_v, rough) * geometry_schlick_ggx(n_dot_l, rough)


@wp.func
def shade_pbr(n: wp.vec3, v: wp.vec3, l: wp.vec3, albedo: wp.vec3,
              rough: float, metallic: float, light_color: wp.vec3) -> wp.vec3:
    """Direct radiance from one light. n,v,l unit; v toward eye, l toward light."""
    h = wp.normalize(v + l)
    n_dot_v = wp.max(wp.dot(n, v), 1e-4)
    n_dot_l = wp.max(wp.dot(n, l), 0.0)
    n_dot_h = wp.max(wp.dot(n, h), 0.0)
    v_dot_h = wp.max(wp.dot(v, h), 0.0)

    f0 = albedo * metallic + wp.vec3(0.04, 0.04, 0.04) * (1.0 - metallic)
    dd = distribution_ggx(n_dot_h, rough)
    gg = geometry_smith(n_dot_v, n_dot_l, rough)
    ff = fresnel_schlick(v_dot_h, f0)

    spec = ff * (dd * gg / (4.0 * n_dot_v * n_dot_l + 1e-4))
    kd = (wp.vec3(1.0, 1.0, 1.0) - ff) * (1.0 - metallic)
    diffuse = wp.cw_mul(kd, albedo) * (1.0 / _PI)
    return wp.cw_mul(diffuse + spec, light_color) * n_dot_l
