"""Signed-distance-field primitives and operators (Warp device functions).

Ports the canonical IQ SDF set (iquilezles.org/articles/distfunctions) plus the
boolean/rounding operators used in the-virus-block-mc's `sdf_library.glsl`. A
scene defines its own `map(p)` by composing these; the P2 raymarcher then sphere-
traces it and derives normals from the gradient of that `map`.
"""

import warp as wp


# ---- primitives (return signed distance; <0 inside) ------------------------

@wp.func
def sd_sphere(p: wp.vec3, r: float) -> float:
    return wp.length(p) - r


@wp.func
def sd_box(p: wp.vec3, b: wp.vec3) -> float:
    d = wp.vec3(wp.abs(p[0]) - b[0], wp.abs(p[1]) - b[1], wp.abs(p[2]) - b[2])
    outside = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0)))
    inside = wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)
    return outside + inside


@wp.func
def sd_round_box(p: wp.vec3, b: wp.vec3, r: float) -> float:
    return sd_box(p, b) - r


@wp.func
def sd_torus(p: wp.vec3, t: wp.vec2) -> float:
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - t[0], p[1])
    return wp.length(q) - t[1]


@wp.func
def sd_cylinder(p: wp.vec3, h: float, r: float) -> float:
    dxz = wp.length(wp.vec2(p[0], p[2])) - r
    dy = wp.abs(p[1]) - h
    outside = wp.length(wp.vec2(wp.max(dxz, 0.0), wp.max(dy, 0.0)))
    inside = wp.min(wp.max(dxz, dy), 0.0)
    return outside + inside


@wp.func
def sd_capsule(p: wp.vec3, a: wp.vec3, b: wp.vec3, r: float) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h) - r


@wp.func
def sd_plane(p: wp.vec3, n: wp.vec3, h: float) -> float:
    return wp.dot(p, n) + h


@wp.func
def sd_ellipsoid(p: wp.vec3, r: wp.vec3) -> float:
    k0 = wp.length(wp.vec3(p[0] / r[0], p[1] / r[1], p[2] / r[2]))
    k1 = wp.length(wp.vec3(p[0] / (r[0] * r[0]), p[1] / (r[1] * r[1]), p[2] / (r[2] * r[2])))
    return k0 * (k0 - 1.0) / wp.max(k1, 1e-8)


# ---- operators -------------------------------------------------------------

@wp.func
def op_union(a: float, b: float) -> float:
    return wp.min(a, b)


@wp.func
def op_intersect(a: float, b: float) -> float:
    return wp.max(a, b)


@wp.func
def op_subtract(a: float, b: float) -> float:
    return wp.max(a, -b)


@wp.func
def op_smooth_union(a: float, b: float, k: float) -> float:
    h = wp.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return wp.lerp(b, a, h) - k * h * (1.0 - h)


@wp.func
def op_smooth_subtract(a: float, b: float, k: float) -> float:
    h = wp.clamp(0.5 - 0.5 * (a + b) / k, 0.0, 1.0)
    return wp.lerp(a, -b, h) + k * h * (1.0 - h)


@wp.func
def op_smooth_intersect(a: float, b: float, k: float) -> float:
    h = wp.clamp(0.5 - 0.5 * (b - a) / k, 0.0, 1.0)
    return wp.lerp(b, a, h) + k * h * (1.0 - h)


@wp.func
def op_round(d: float, r: float) -> float:
    return d - r


@wp.func
def op_onion(d: float, thickness: float) -> float:
    return wp.abs(d) - thickness
