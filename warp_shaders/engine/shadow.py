"""Analytic soft shadows + ambient occlusion — reusable ``@wp.func`` primitives.

Most raymarchers compute soft shadows and AO by re-marching the scene SDF, which
cannot be factored into a shared device function (Warp can't pass a callable).
But for the very common case of a **sphere occluder** — planets, moons, stars,
the sphere-garden demo — both have exact closed forms with *no* SDF callback, so
they *are* shareable. Call them from your own kernel:

- :func:`soft_shadow_sphere` — penumbra soft shadow of a sphere along a light ray
  (Quilez, "sphere functions"). Returns visibility in ``[0, 1]``.
- :func:`sphere_occlusion` / :func:`sphere_ao` — exact analytic ambient occlusion
  of a hemisphere by a sphere (Quilez). ``sphere_ao`` returns *visibility* so it
  composes multiplicatively just like the shadow.
- :func:`penumbra` — the running-min accumulation atom of the IQ *SDF* soft
  shadow, so heightfield / SDF marchers share the one constant instead of
  copy-pasting ``res = min(res, k*h/t)``.
- :func:`ground_contact_ao` — cheap contact darkening for a subject a height `h`
  above a plane (the billboard-on-ground case).

Sources (docs/research): Inigo Quilez, "sphere functions"
(iquilezles.org/articles/spherefunctions) — analytic ``sphSoftShadow`` +
``sphOcclusion``; the SDF penumbra term is his classic soft-shadow marcher.
"""

from __future__ import annotations

import warp as wp

_PI = wp.constant(3.14159265358979)


@wp.func
def soft_shadow_sphere(p: wp.vec3, l: wp.vec3, ce: wp.vec3, ra: float,
                       k: float) -> float:
    """Visibility of a light in direction `l` (unit) at surface point `p`, cast
    against a sphere occluder (`ce`, `ra`). ``1`` = lit, ``0`` = fully shadowed,
    with a penumbra whose hardness grows with `k` (typical 4..32).

    Exact geometry: ``d = perp_distance - radius`` is the angular gap past the
    sphere's edge; dividing by the forward distance to closest approach turns it
    into a subtended-angle penumbra. A hit in front (``d<0, t>0``) is full shadow;
    an occluder behind the point (``t<0``) casts nothing."""
    oc = p - ce
    b = wp.dot(oc, l)
    c = wp.dot(oc, oc) - ra * ra
    h = b * b - c
    # sqrt(max(0, ra^2 - h)) == perpendicular distance from the sphere centre to
    # the ray; subtract the radius for the signed edge gap.
    d = -ra + wp.sqrt(wp.max(ra * ra - h, 0.0))
    t = -b - wp.sqrt(wp.max(h, 0.0))
    if t < 0.0:
        return 1.0
    return wp.clamp(k * d / t, 0.0, 1.0)


@wp.func
def sphere_occlusion(p: wp.vec3, n: wp.vec3, ce: wp.vec3, ra: float) -> float:
    """Exact analytic ambient occlusion of the hemisphere at `p` (normal `n`) by
    a sphere (`ce`, `ra`). Returns the **occluded fraction** in ``[0, 1]``
    (0 = sky fully visible). Quilez's closed form: a fast horizon approximation
    when the sphere is entirely above/below the horizon, and the exact
    partial-horizon integral otherwise."""
    di = ce - p
    l = wp.length(di)
    if l < 1.0e-6:
        return 1.0
    nl = wp.dot(n, di * (1.0 / l))
    h = l / wp.max(ra, 1.0e-6)
    h2 = h * h
    k2 = 1.0 - h2 * nl * nl
    res = wp.max(nl, 0.0) / h2                       # sphere above/below horizon
    if k2 > 0.001 and h2 > 1.0:                      # sphere crosses the horizon
        num = wp.sqrt((h2 - 1.0) / wp.max(1.0 - nl * nl, 1.0e-6))
        res = nl * wp.acos(wp.clamp(-nl * num, -1.0, 1.0)) - wp.sqrt(k2 * (h2 - 1.0))
        res = (res / h2 + wp.atan(wp.sqrt(k2 / (h2 - 1.0)))) / _PI
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def sphere_ao(p: wp.vec3, n: wp.vec3, ce: wp.vec3, ra: float) -> float:
    """Ambient-occlusion **visibility** at `p` from a sphere occluder — the
    complement of :func:`sphere_occlusion`, so it multiplies into ambient light
    exactly like a shadow term (``1`` = unoccluded)."""
    return 1.0 - sphere_occlusion(p, n, ce, ra)


@wp.func
def penumbra(res: float, h: float, t: float, k: float) -> float:
    """The IQ *SDF* soft-shadow accumulation atom — call once per shadow-march
    step with the current SDF distance `h` at travelled distance `t`:
    ``res = min(res, k*h/t)``. Seed `res` at ``1`` before the loop, clamp after.
    Sharing this keeps every SDF marcher on the same penumbra constant."""
    return wp.min(res, k * h / wp.max(t, 1.0e-4))


@wp.func
def ground_contact_ao(height: float, radius: float) -> float:
    """Cheap contact darkening for a subject `height` above a ground plane,
    softening over `radius`. Returns visibility in ``[0, 1]`` — 0 right at the
    contact point, rising to 1 well above the plane (billboard-on-ground case)."""
    return wp.clamp(1.0 - wp.exp(-height / wp.max(radius, 1.0e-4)), 0.0, 1.0)
