"""Reflection + refraction primitives for the raymarcher (device ``@wp.func``).

The engine's SDF scenes march a primary ray; these atoms let a hit **bounce**:
mirror reflection, dielectric refraction (Snell, with total-internal-reflection),
and the Fresnel term that blends the two by angle. They're plain vector maths — no
SDF callback — so they compile into any kernel; the multi-bounce *loop* lives in
the scene (Warp has no recursion), see ``scenes/reflections.py``.

Conventions: ``i`` is the **incident** ray direction (unit, pointing *toward* the
surface); ``n`` is the surface normal (unit, pointing *against* the incoming ray,
i.e. outward). ``eta`` is the ratio of indices of refraction ``n1 / n2`` across
the interface (air→glass ≈ 1/1.5, glass→air ≈ 1.5).

Sources (docs/research): Schlick 1994 (Fresnel approximation); Snell's law;
Whitted 1980 (recursive ray tracing) — realised here as a bounded bounce loop.
"""

from __future__ import annotations

import warp as wp


@wp.func
def reflect(i: wp.vec3, n: wp.vec3) -> wp.vec3:
    """Mirror-reflect incident direction `i` about surface normal `n`."""
    return i - n * (2.0 * wp.dot(i, n))


@wp.func
def refract_k(i: wp.vec3, n: wp.vec3, eta: float) -> float:
    """Snell discriminant ``k``; ``k < 0`` means total internal reflection."""
    ci = -wp.dot(i, n)
    return 1.0 - eta * eta * (1.0 - ci * ci)


@wp.func
def refract(i: wp.vec3, n: wp.vec3, eta: float) -> wp.vec3:
    """Refract `i` through the interface (Snell). On total internal reflection
    (``refract_k < 0``) there is no transmitted ray, so this returns the mirror
    reflection instead — the physically correct fallback."""
    ci = -wp.dot(i, n)
    k = 1.0 - eta * eta * (1.0 - ci * ci)
    if k < 0.0:
        return i - n * (2.0 * wp.dot(i, n))            # TIR -> reflect
    return i * eta + n * (eta * ci - wp.sqrt(k))


@wp.func
def schlick_f0(ior: float) -> float:
    """Normal-incidence reflectance ``F0`` of a dielectric from its index."""
    r = (ior - 1.0) / (ior + 1.0)
    return r * r


@wp.func
def fresnel_dielectric(cos_theta: float, ior: float) -> float:
    """Schlick Fresnel reflectance of a dielectric at incidence angle `cos_theta`
    (``= |n·v|``). Goes from ``F0`` head-on to ``1`` at grazing."""
    f0 = schlick_f0(ior)
    c = wp.clamp(1.0 - wp.max(cos_theta, 0.0), 0.0, 1.0)
    c2 = c * c
    return f0 + (1.0 - f0) * (c2 * c2 * c)             # (1-cos)^5


@wp.func
def fresnel_schlick_s(cos_theta: float, f0: float) -> float:
    """Scalar Schlick Fresnel from an explicit ``F0`` (for metals/tinted mirrors)."""
    c = wp.clamp(1.0 - wp.max(cos_theta, 0.0), 0.0, 1.0)
    c2 = c * c
    return f0 + (1.0 - f0) * (c2 * c2 * c)
