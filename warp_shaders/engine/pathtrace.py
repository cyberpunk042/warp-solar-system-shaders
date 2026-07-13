"""Monte-Carlo path-tracing helpers (device ``@wp.func`` + host camera math).

The sampling primitives shared by the path-traced scenes (``cornell_box``, ``glass_box``,
``subsurface``, ``motion_blur``): cosine-weighted hemisphere sampling for diffuse bounces,
uniform sphere sampling for isotropic scatter, and a host-side camera-basis helper. Specular
BSDF primitives (reflect / refract / Fresnel) live in :mod:`engine.raytrace`; this module is the
importance-sampling side of the same integrator.
"""

import math

import numpy as np
import warp as wp

_TWO_PI = wp.constant(6.2831853071795864)


@wp.func
def onb_cosine(n: wp.vec3, r1: float, r2: float) -> wp.vec3:
    """A cosine-weighted direction in the hemisphere about `n` (importance-samples the
    Lambertian BRDF, so the cosine and the pdf cancel — the estimator is just ``*= albedo``)."""
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    tang = wp.normalize(wp.cross(a, n))
    bit = wp.cross(n, tang)
    r = wp.sqrt(r1)
    phi = _TWO_PI * r2
    return wp.normalize(tang * (r * wp.cos(phi)) + bit * (r * wp.sin(phi))
                        + n * wp.sqrt(wp.max(0.0, 1.0 - r1)))


@wp.func
def sample_sphere(r1: float, r2: float) -> wp.vec3:
    """A uniformly-distributed direction on the unit sphere (isotropic scatter)."""
    z = 1.0 - 2.0 * r1
    r = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    phi = _TWO_PI * r2
    return wp.vec3(r * wp.cos(phi), r * wp.sin(phi), z)


def camera_basis(eye, tgt, up=(0.0, 1.0, 0.0)):
    """Return ``(fwd, right, up)`` unit vectors for a look-at camera, as ``wp.vec3``."""
    eye = np.asarray(eye, np.float32)
    fwd = np.asarray(tgt, np.float32) - eye
    fwd /= np.linalg.norm(fwd) + 1e-9
    right = np.cross(fwd, np.asarray(up, np.float32))
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd)
    return (wp.vec3(*[float(x) for x in fwd]),
            wp.vec3(*[float(x) for x in right]),
            wp.vec3(*[float(x) for x in upv]))


def tanfov(fov_deg):
    return math.tan(math.radians(fov_deg) * 0.5)
