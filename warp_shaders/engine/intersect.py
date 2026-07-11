"""Ray-primitive intersection — the shared analytic tests every scene needs.

Consolidates the ray-sphere helper that was copy-pasted (as ``_rs`` / ``_ray_sphere``
/ inline) across a dozen scenes. All return the same convention as the originals
so adoption is a drop-in:

- `ray_sphere` / `ray_sphere_o` → `wp.vec2(t_near, t_far)`; a **miss** returns
  ``(1e30, -1e30)`` (so ``hit[1] > hit[0]`` ⇔ hit, and ``hit[1] > 0`` ⇔ in front).
- `ray_box` (AABB slab) → same `(t_near, t_far)` convention.
- `ray_plane` / `ray_disk` → a single `t` (or ``-1`` on miss / behind).
- `sphere_t` → the nearest positive hit distance (or ``-1``), the common case.
"""

from __future__ import annotations

import warp as wp

_MISS = wp.constant(1.0e30)


@wp.func
def ray_sphere(ro: wp.vec3, rd: wp.vec3, center: wp.vec3,
               radius: float) -> wp.vec2:
    """Ray vs a sphere at `center`. Returns (t_near, t_far); miss → (1e30,-1e30)."""
    oc = ro - center
    b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return wp.vec2(1.0e30, -1.0e30)
    s = wp.sqrt(disc)
    return wp.vec2(-b - s, -b + s)


@wp.func
def ray_sphere_o(ro: wp.vec3, rd: wp.vec3, radius: float) -> wp.vec2:
    """Ray vs an origin-centred sphere. Returns (t_near, t_far); miss → (1e30,-1e30)."""
    b = wp.dot(ro, rd)
    c = wp.dot(ro, ro) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return wp.vec2(1.0e30, -1.0e30)
    s = wp.sqrt(disc)
    return wp.vec2(-b - s, -b + s)


@wp.func
def sphere_t(ro: wp.vec3, rd: wp.vec3, center: wp.vec3, radius: float) -> float:
    """Nearest positive intersection distance with a sphere, or -1 on miss."""
    h = ray_sphere(ro, rd, center, radius)
    if h[1] < h[0] or h[1] <= 0.0:
        return -1.0
    if h[0] > 0.0:
        return h[0]
    return h[1]                                  # inside the sphere → far root


@wp.func
def ray_plane(ro: wp.vec3, rd: wp.vec3, p0: wp.vec3, n: wp.vec3) -> float:
    """Ray vs an infinite plane through `p0` with normal `n`. t, or -1."""
    dn = wp.dot(rd, n)
    if wp.abs(dn) < 1.0e-8:
        return -1.0
    t = wp.dot(p0 - ro, n) / dn
    if t < 0.0:
        return -1.0
    return t


@wp.func
def ray_disk(ro: wp.vec3, rd: wp.vec3, center: wp.vec3, n: wp.vec3,
             radius: float) -> float:
    """Ray vs a flat disk (plane through `center`, normal `n`, radius). t, or -1."""
    t = ray_plane(ro, rd, center, n)
    if t < 0.0:
        return -1.0
    p = ro + rd * t
    if wp.length(p - center) > radius:
        return -1.0
    return t


@wp.func
def ray_box(ro: wp.vec3, rd: wp.vec3, bmin: wp.vec3, bmax: wp.vec3) -> wp.vec2:
    """Ray vs an axis-aligned box (slab method). Returns (t_near, t_far); a miss
    has ``t_near > t_far``."""
    inv = wp.vec3(1.0 / rd[0], 1.0 / rd[1], 1.0 / rd[2])
    t0 = wp.cw_mul(bmin - ro, inv)
    t1 = wp.cw_mul(bmax - ro, inv)
    tmin = wp.min(t0, t1)
    tmax = wp.max(t0, t1)
    tn = wp.max(wp.max(tmin[0], tmin[1]), tmin[2])
    tf = wp.min(wp.min(tmax[0], tmax[1]), tmax[2])
    return wp.vec2(tn, tf)
