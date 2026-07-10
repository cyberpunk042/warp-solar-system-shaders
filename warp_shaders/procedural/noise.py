"""Procedural noise toolkit (Warp device functions).

Value & Perlin gradient noise (with analytic derivatives), Worley/cellular noise,
and the fbm / ridged / billow / domain-warp / curl composites built on them.

Sources (see docs/research/00-foundations.md):
- Value / gradient noise + derivatives: Inigo Quilez (iquilezles.org/articles/morenoise,
  /gradientnoise) — quintic interpolation.
- Worley / cellular: Steven Worley 1996 (F1/F2 over a 3x3x3 neighborhood).
- fbm / domain warp: Inigo Quilez (iquilezles.org/articles/fbm, /warp).
- Curl noise: Bridson et al. 2007 (curl of a noise potential; divergence-free).

Conventions: `value3`/`worley3` return ~[0,1]; `perlin3`/`simplex-like` return ~[-1,1];
`fbm3` returns ~[0,1].
"""

import warp as wp

from .hash import hash31, hash33


@wp.func
def _quintic(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


@wp.func
def _dquintic(t: float) -> float:
    return 30.0 * t * t * (t * (t - 2.0) + 1.0)


# ---- value noise -----------------------------------------------------------

@wp.func
def value3(p: wp.vec3) -> float:
    """Value noise in [0,1] with quintic interpolation."""
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - i
    u = wp.vec3(_quintic(f[0]), _quintic(f[1]), _quintic(f[2]))

    a = hash31(i + wp.vec3(0.0, 0.0, 0.0))
    b = hash31(i + wp.vec3(1.0, 0.0, 0.0))
    c = hash31(i + wp.vec3(0.0, 1.0, 0.0))
    d = hash31(i + wp.vec3(1.0, 1.0, 0.0))
    e = hash31(i + wp.vec3(0.0, 0.0, 1.0))
    g = hash31(i + wp.vec3(1.0, 0.0, 1.0))
    h = hash31(i + wp.vec3(0.0, 1.0, 1.0))
    k = hash31(i + wp.vec3(1.0, 1.0, 1.0))

    ab = wp.lerp(a, b, u[0])
    cd = wp.lerp(c, d, u[0])
    eg = wp.lerp(e, g, u[0])
    hk = wp.lerp(h, k, u[0])
    abcd = wp.lerp(ab, cd, u[1])
    eghk = wp.lerp(eg, hk, u[1])
    return wp.lerp(abcd, eghk, u[2])


@wp.func
def noised3(p: wp.vec3) -> wp.vec4:
    """Value noise + analytic gradient: returns vec4(value, d/dx, d/dy, d/dz).

    Lets surfaces compute normals without finite differences (IQ 'morenoise')."""
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - i
    u = wp.vec3(_quintic(f[0]), _quintic(f[1]), _quintic(f[2]))
    du = wp.vec3(_dquintic(f[0]), _dquintic(f[1]), _dquintic(f[2]))

    a = hash31(i + wp.vec3(0.0, 0.0, 0.0))
    b = hash31(i + wp.vec3(1.0, 0.0, 0.0))
    c = hash31(i + wp.vec3(0.0, 1.0, 0.0))
    d = hash31(i + wp.vec3(1.0, 1.0, 0.0))
    e = hash31(i + wp.vec3(0.0, 0.0, 1.0))
    g = hash31(i + wp.vec3(1.0, 0.0, 1.0))
    h = hash31(i + wp.vec3(0.0, 1.0, 1.0))
    k = hash31(i + wp.vec3(1.0, 1.0, 1.0))

    # coefficients of the trilinear form
    k0 = a
    k1 = b - a
    k2 = c - a
    k3 = e - a
    k4 = a - b - c + d
    k5 = a - c - e + h
    k6 = a - b - e + g
    k7 = -a + b + c - d + e - g - h + k

    ux = u[0]
    uy = u[1]
    uz = u[2]
    val = (k0 + k1 * ux + k2 * uy + k3 * uz + k4 * ux * uy
           + k5 * uy * uz + k6 * uz * ux + k7 * ux * uy * uz)
    dx = du[0] * (k1 + k4 * uy + k6 * uz + k7 * uy * uz)
    dy = du[1] * (k2 + k5 * uz + k4 * ux + k7 * uz * ux)
    dz = du[2] * (k3 + k6 * ux + k5 * uy + k7 * ux * uy)
    return wp.vec4(val, dx, dy, dz)


# ---- gradient (Perlin) noise ----------------------------------------------

@wp.func
def _grad(icorner: wp.vec3, d: wp.vec3) -> float:
    g = hash33(icorner) * 2.0 - wp.vec3(1.0, 1.0, 1.0)
    return wp.dot(g, d)


@wp.func
def perlin3(p: wp.vec3) -> float:
    """Gradient (Perlin) noise, ~[-1,1]."""
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - i
    u = wp.vec3(_quintic(f[0]), _quintic(f[1]), _quintic(f[2]))

    n000 = _grad(i + wp.vec3(0.0, 0.0, 0.0), f - wp.vec3(0.0, 0.0, 0.0))
    n100 = _grad(i + wp.vec3(1.0, 0.0, 0.0), f - wp.vec3(1.0, 0.0, 0.0))
    n010 = _grad(i + wp.vec3(0.0, 1.0, 0.0), f - wp.vec3(0.0, 1.0, 0.0))
    n110 = _grad(i + wp.vec3(1.0, 1.0, 0.0), f - wp.vec3(1.0, 1.0, 0.0))
    n001 = _grad(i + wp.vec3(0.0, 0.0, 1.0), f - wp.vec3(0.0, 0.0, 1.0))
    n101 = _grad(i + wp.vec3(1.0, 0.0, 1.0), f - wp.vec3(1.0, 0.0, 1.0))
    n011 = _grad(i + wp.vec3(0.0, 1.0, 1.0), f - wp.vec3(0.0, 1.0, 1.0))
    n111 = _grad(i + wp.vec3(1.0, 1.0, 1.0), f - wp.vec3(1.0, 1.0, 1.0))

    nx00 = wp.lerp(n000, n100, u[0])
    nx10 = wp.lerp(n010, n110, u[0])
    nx01 = wp.lerp(n001, n101, u[0])
    nx11 = wp.lerp(n011, n111, u[0])
    nxy0 = wp.lerp(nx00, nx10, u[1])
    nxy1 = wp.lerp(nx01, nx11, u[1])
    return wp.lerp(nxy0, nxy1, u[2]) * 1.15


# ---- Worley / cellular -----------------------------------------------------

@wp.func
def worley3(p: wp.vec3) -> float:
    """Cellular noise: distance to the nearest feature point (F1), ~[0,1]."""
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - i
    f1 = float(1e9)
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                g = wp.vec3(float(x), float(y), float(z))
                o = hash33(i + g)
                d = wp.length(g + o - f)
                f1 = wp.min(f1, d)
    return f1


@wp.func
def worley3_f2(p: wp.vec3) -> wp.vec2:
    """Returns (F1, F2): nearest and second-nearest feature distances."""
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - i
    f1 = float(1e9)
    f2 = float(1e9)
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                g = wp.vec3(float(x), float(y), float(z))
                o = hash33(i + g)
                d = wp.length(g + o - f)
                if d < f1:
                    f2 = f1
                    f1 = d
                else:
                    if d < f2:
                        f2 = d
    return wp.vec2(f1, f2)


# ---- fbm & composites ------------------------------------------------------

@wp.func
def fbm3(p: wp.vec3, octaves: int) -> float:
    """Fractal Brownian motion over value noise, normalized to ~[0,1]."""
    v = float(0.0)
    amp = float(0.5)
    freq = float(1.0)
    norm = float(0.0)
    for _ in range(octaves):
        v += amp * value3(p * freq)
        norm += amp
        freq *= 2.0
        amp *= 0.5
    return v / wp.max(norm, 1e-6)


@wp.func
def fbm_perlin3(p: wp.vec3, octaves: int) -> float:
    """fBM over Perlin gradient noise, ~[-1,1]."""
    v = float(0.0)
    amp = float(0.5)
    freq = float(1.0)
    for _ in range(octaves):
        v += amp * perlin3(p * freq)
        freq *= 2.0
        amp *= 0.5
    return v


@wp.func
def ridged3(p: wp.vec3, octaves: int) -> float:
    """Ridged multifractal (sharp crests) — good for mountains/lightning."""
    v = float(0.0)
    amp = float(0.5)
    freq = float(1.0)
    for _ in range(octaves):
        n = 1.0 - wp.abs(perlin3(p * freq))
        v += amp * n * n
        freq *= 2.0
        amp *= 0.5
    return v


@wp.func
def billow3(p: wp.vec3, octaves: int) -> float:
    """Billow (puffy) fractal — good for clouds."""
    v = float(0.0)
    amp = float(0.5)
    freq = float(1.0)
    norm = float(0.0)
    for _ in range(octaves):
        v += amp * wp.abs(perlin3(p * freq))
        norm += amp
        freq *= 2.0
        amp *= 0.5
    return v / wp.max(norm, 1e-6)


@wp.func
def domain_warp3(p: wp.vec3, octaves: int, amount: float) -> float:
    """fBM of coordinates warped by fBM — organic, flowing patterns (IQ 'warp')."""
    q = wp.vec3(fbm3(p + wp.vec3(0.0, 0.0, 0.0), octaves),
                fbm3(p + wp.vec3(5.2, 1.3, 2.7), octaves),
                fbm3(p + wp.vec3(2.7, 8.3, 4.1), octaves))
    return fbm3(p + (q * 2.0 - wp.vec3(1.0, 1.0, 1.0)) * amount, octaves)


@wp.func
def curl3(p: wp.vec3) -> wp.vec3:
    """Divergence-free flow field = curl of a noise potential (Bridson 2007)."""
    e = float(0.1)
    # three independent potential fields via large offsets
    p1 = p
    p2 = p + wp.vec3(31.4, 0.0, 0.0)
    p3 = p + wp.vec3(0.0, 47.2, 0.0)
    dx = wp.vec3(e, 0.0, 0.0)
    dy = wp.vec3(0.0, e, 0.0)
    dz = wp.vec3(0.0, 0.0, e)
    # partials of each potential
    p1_dy = (perlin3(p1 + dy) - perlin3(p1 - dy)) / (2.0 * e)
    p1_dz = (perlin3(p1 + dz) - perlin3(p1 - dz)) / (2.0 * e)
    p2_dx = (perlin3(p2 + dx) - perlin3(p2 - dx)) / (2.0 * e)
    p2_dz = (perlin3(p2 + dz) - perlin3(p2 - dz)) / (2.0 * e)
    p3_dx = (perlin3(p3 + dx) - perlin3(p3 - dx)) / (2.0 * e)
    p3_dy = (perlin3(p3 + dy) - perlin3(p3 - dy)) / (2.0 * e)
    return wp.vec3(p3_dy - p2_dz, p1_dz - p3_dx, p2_dx - p1_dy)
