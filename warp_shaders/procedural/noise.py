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
def _floor4(v: wp.vec4) -> wp.vec4:
    return wp.vec4(wp.floor(v[0]), wp.floor(v[1]), wp.floor(v[2]), wp.floor(v[3]))


@wp.func
def _stepf(edge: float, x: float) -> float:
    if x >= edge:
        return 1.0
    return 0.0


@wp.func
def _mod289_3(x: wp.vec3) -> wp.vec3:
    return x - wp.vec3(wp.floor(x[0] / 289.0), wp.floor(x[1] / 289.0), wp.floor(x[2] / 289.0)) * 289.0


@wp.func
def _mod289_4(x: wp.vec4) -> wp.vec4:
    return x - _floor4(x * (1.0 / 289.0)) * 289.0


@wp.func
def _permute(x: wp.vec4) -> wp.vec4:
    return _mod289_4(wp.cw_mul(x * 34.0 + wp.vec4(1.0, 1.0, 1.0, 1.0), x))


@wp.func
def _taylor_inv_sqrt(r: wp.vec4) -> wp.vec4:
    return wp.vec4(1.79284291400159, 1.79284291400159, 1.79284291400159, 1.79284291400159) - r * 0.85373472095314


@wp.func
def simplex3(v: wp.vec3) -> float:
    """Perlin simplex noise (Gustavson/Ashima port), ~[-1,1]. Fewer directional
    artifacts than value/Perlin noise."""
    cx = 0.166666666667
    cy = 0.333333333333
    i = wp.vec3(wp.floor(v[0] + wp.dot(v, wp.vec3(cy, cy, cy))),
                wp.floor(v[1] + wp.dot(v, wp.vec3(cy, cy, cy))),
                wp.floor(v[2] + wp.dot(v, wp.vec3(cy, cy, cy))))
    x0 = v - i + wp.vec3(wp.dot(i, wp.vec3(cx, cx, cx)), wp.dot(i, wp.vec3(cx, cx, cx)),
                         wp.dot(i, wp.vec3(cx, cx, cx)))

    g = wp.vec3(_stepf(x0[1], x0[0]), _stepf(x0[2], x0[1]), _stepf(x0[0], x0[2]))
    l = wp.vec3(1.0, 1.0, 1.0) - g
    lzxy = wp.vec3(l[2], l[0], l[1])
    i1 = wp.vec3(wp.min(g[0], lzxy[0]), wp.min(g[1], lzxy[1]), wp.min(g[2], lzxy[2]))
    i2 = wp.vec3(wp.max(g[0], lzxy[0]), wp.max(g[1], lzxy[1]), wp.max(g[2], lzxy[2]))

    x1 = x0 - i1 + wp.vec3(cx, cx, cx)
    x2 = x0 - i2 + wp.vec3(cy, cy, cy)
    x3 = x0 - wp.vec3(0.5, 0.5, 0.5)

    i = _mod289_3(i)
    p = _permute(_permute(_permute(
        wp.vec4(i[2] + 0.0, i[2] + i1[2], i[2] + i2[2], i[2] + 1.0))
        + wp.vec4(i[1] + 0.0, i[1] + i1[1], i[1] + i2[1], i[1] + 1.0))
        + wp.vec4(i[0] + 0.0, i[0] + i1[0], i[0] + i2[0], i[0] + 1.0))

    ns_x = 0.285714285714
    ns_y = -0.928571428571
    ns_z = 0.142857142857
    j = p - 49.0 * _floor4(p * (ns_z * ns_z))
    x_ = _floor4(j * ns_z)
    y_ = _floor4(j - 7.0 * x_)
    xx = x_ * ns_x + wp.vec4(ns_y, ns_y, ns_y, ns_y)
    yy = y_ * ns_x + wp.vec4(ns_y, ns_y, ns_y, ns_y)
    h = wp.vec4(1.0, 1.0, 1.0, 1.0) - wp.vec4(wp.abs(xx[0]), wp.abs(xx[1]), wp.abs(xx[2]), wp.abs(xx[3])) \
        - wp.vec4(wp.abs(yy[0]), wp.abs(yy[1]), wp.abs(yy[2]), wp.abs(yy[3]))

    b0 = wp.vec4(xx[0], xx[1], yy[0], yy[1])
    b1 = wp.vec4(xx[2], xx[3], yy[2], yy[3])
    s0 = _floor4(b0) * 2.0 + wp.vec4(1.0, 1.0, 1.0, 1.0)
    s1 = _floor4(b1) * 2.0 + wp.vec4(1.0, 1.0, 1.0, 1.0)
    # sh = -step(h, 0): -1 where h < 0, else 0
    sh0 = float(0.0)
    if h[0] < 0.0:
        sh0 = -1.0
    sh1 = float(0.0)
    if h[1] < 0.0:
        sh1 = -1.0
    sh2 = float(0.0)
    if h[2] < 0.0:
        sh2 = -1.0
    sh3 = float(0.0)
    if h[3] < 0.0:
        sh3 = -1.0

    a0 = wp.vec4(b0[0] + s0[0] * sh0, b0[2] + s0[2] * sh0, b0[1] + s0[1] * sh1, b0[3] + s0[3] * sh1)
    a1 = wp.vec4(b1[0] + s1[0] * sh2, b1[2] + s1[2] * sh2, b1[1] + s1[1] * sh3, b1[3] + s1[3] * sh3)

    p0 = wp.vec3(a0[0], a0[1], h[0])
    p1 = wp.vec3(a0[2], a0[3], h[1])
    p2 = wp.vec3(a1[0], a1[1], h[2])
    p3 = wp.vec3(a1[2], a1[3], h[3])

    norm = _taylor_inv_sqrt(wp.vec4(wp.dot(p0, p0), wp.dot(p1, p1), wp.dot(p2, p2), wp.dot(p3, p3)))
    p0 = p0 * norm[0]
    p1 = p1 * norm[1]
    p2 = p2 * norm[2]
    p3 = p3 * norm[3]

    m = wp.vec4(wp.max(0.6 - wp.dot(x0, x0), 0.0), wp.max(0.6 - wp.dot(x1, x1), 0.0),
                wp.max(0.6 - wp.dot(x2, x2), 0.0), wp.max(0.6 - wp.dot(x3, x3), 0.0))
    m = wp.cw_mul(m, m)
    m = wp.cw_mul(m, m)
    return 42.0 * wp.dot(m, wp.vec4(wp.dot(p0, x0), wp.dot(p1, x1), wp.dot(p2, x2), wp.dot(p3, x3)))


@wp.func
def _modp(x: float, m: float) -> float:
    return x - wp.floor(x / m) * m


@wp.func
def value_tiled3(p: wp.vec3, period: float) -> float:
    """Value noise that tiles seamlessly with the given integer period."""
    i = wp.vec3(wp.floor(p[0]), wp.floor(p[1]), wp.floor(p[2]))
    f = p - i
    u = wp.vec3(_quintic(f[0]), _quintic(f[1]), _quintic(f[2]))

    m = period
    a = hash31(wp.vec3(_modp(i[0], m), _modp(i[1], m), _modp(i[2], m)))
    b = hash31(wp.vec3(_modp(i[0] + 1.0, m), _modp(i[1], m), _modp(i[2], m)))
    c = hash31(wp.vec3(_modp(i[0], m), _modp(i[1] + 1.0, m), _modp(i[2], m)))
    d = hash31(wp.vec3(_modp(i[0] + 1.0, m), _modp(i[1] + 1.0, m), _modp(i[2], m)))
    e = hash31(wp.vec3(_modp(i[0], m), _modp(i[1], m), _modp(i[2] + 1.0, m)))
    g = hash31(wp.vec3(_modp(i[0] + 1.0, m), _modp(i[1], m), _modp(i[2] + 1.0, m)))
    hh = hash31(wp.vec3(_modp(i[0], m), _modp(i[1] + 1.0, m), _modp(i[2] + 1.0, m)))
    k = hash31(wp.vec3(_modp(i[0] + 1.0, m), _modp(i[1] + 1.0, m), _modp(i[2] + 1.0, m)))

    ab = wp.lerp(a, b, u[0])
    cd = wp.lerp(c, d, u[0])
    eg = wp.lerp(e, g, u[0])
    hk = wp.lerp(hh, k, u[0])
    return wp.lerp(wp.lerp(ab, cd, u[1]), wp.lerp(eg, hk, u[1]), u[2])


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
