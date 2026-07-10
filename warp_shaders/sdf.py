"""Reusable Warp device functions for procedural raymarching.

These are the building blocks — hashing, value noise, fBm, 2D rotation, and a
couple of signed-distance primitives — ported from the usual GLSL idioms into
NVIDIA Warp ``@wp.func`` form. Import them into a scene module and compose them
inside a ``@wp.kernel``. Everything runs identically on CPU and CUDA.

GLSL has no scalar/vector distinction at the syntax level and supports swizzles
(``p.xz = ...``); Warp does not, so vectors are rebuilt component-wise. The math
is otherwise a direct translation.
"""

import warp as wp


@wp.func
def fract(x: float) -> float:
    return x - wp.floor(x)


@wp.func
def rot2(p: wp.vec2, a: float) -> wp.vec2:
    """Rotate a 2D vector by ``a`` radians."""
    s = wp.sin(a)
    c = wp.cos(a)
    return wp.vec2(c * p[0] - s * p[1], s * p[0] + c * p[1])


@wp.func
def hash2d(p: wp.vec2) -> float:
    """Cheap hash -> [0, 1). Matches the common ``fract(p*...)`` GLSL hash."""
    p = wp.vec2(fract(p[0] * 123.34), fract(p[1] * 456.21))
    d = p[0] * (p[0] + 45.32) + p[1] * (p[1] + 45.32)  # dot(p, p + 45.32)
    p = wp.vec2(p[0] + d, p[1] + d)
    return fract(p[0] * p[1])


@wp.func
def noise2d(p: wp.vec2) -> float:
    """Value noise with smoothstep interpolation."""
    i = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    f = wp.vec2(p[0] - i[0], p[1] - i[1])
    u = wp.vec2(f[0] * f[0] * (3.0 - 2.0 * f[0]), f[1] * f[1] * (3.0 - 2.0 * f[1]))

    a = hash2d(wp.vec2(i[0] + 0.0, i[1] + 0.0))
    b = hash2d(wp.vec2(i[0] + 1.0, i[1] + 0.0))
    c = hash2d(wp.vec2(i[0] + 0.0, i[1] + 1.0))
    d = hash2d(wp.vec2(i[0] + 1.0, i[1] + 1.0))

    return wp.lerp(wp.lerp(a, b, u[0]), wp.lerp(c, d, u[0]), u[1])


@wp.func
def fbm2d(p: wp.vec2) -> float:
    """3-octave fractional Brownian motion over :func:`noise2d`."""
    value = float(0.0)
    amplitude = float(0.5)
    for _ in range(3):
        value += amplitude * noise2d(p)
        p = wp.vec2(p[0] * 2.5, p[1] * 2.5)
        amplitude *= 0.5
    return value


@wp.func
def sd_torus(p: wp.vec3, t: wp.vec2) -> float:
    """Signed distance to a torus of major/minor radii ``t.x``/``t.y``."""
    q = wp.vec2(wp.length(wp.vec2(p[0], p[2])) - t[0], p[1])
    return wp.length(q) - t[1]
