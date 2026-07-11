"""Distance estimators for 3D escape-time fractals (device ``@wp.func``).

A fractal has no surface equation, but it admits a **distance estimator** — a
lower bound on the distance to the set — which is exactly what the engine's
sphere-tracer needs. Each estimator returns a ``wp.vec4``:

    (de, trap, escape_iter, final_r)

- ``de``    — the distance estimate (march by this; it never overshoots).
- ``trap``  — the **orbit trap** (minimum ``|z|`` over the iteration), a rich
  colour signal (the banded shells).
- ``escape_iter`` / ``final_r`` — where/how the orbit escaped, for tinting.

See ``docs/research/13-3d-fractals.md``. Sources: White & Nylander 2009
(Mandelbulb triplex power), Lowe 2010 (Mandelbox folds), Hvidtfeldt / Quilez
(distance-estimated ray marching + orbit traps).
"""

from __future__ import annotations

import warp as wp


@wp.func
def mandelbulb_de(p: wp.vec3, power: float, iters: int) -> wp.vec4:
    """Distance estimate to the **Mandelbulb** at `p` (classic look at power 8).

    Iterates the triplex power ``z -> z^power + p`` in spherical coordinates and
    tracks the running derivative for the analytic DE ``0.5·log(r)·r/dr``."""
    z = p
    dr = float(1.0)
    r = float(0.0)
    trap = float(1.0e10)
    esc = float(iters)
    for i in range(iters):
        r = wp.length(z)
        if r > 2.0:
            esc = float(i)
            break
        trap = wp.min(trap, r)
        # to spherical, raise radius to the power, multiply the angles by power
        theta = wp.acos(wp.clamp(z[2] / wp.max(r, 1.0e-9), -1.0, 1.0))
        phi = wp.atan2(z[1], z[0])
        dr = wp.pow(r, power - 1.0) * power * dr + 1.0
        zr = wp.pow(r, power)
        theta = theta * power
        phi = phi * power
        st = wp.sin(theta)
        z = wp.vec3(st * wp.cos(phi), st * wp.sin(phi), wp.cos(theta)) * zr + p
    de = 0.5 * wp.log(wp.max(r, 1.0e-9)) * r / wp.max(dr, 1.0e-9)
    return wp.vec4(wp.max(de, 0.0), trap, esc, r)


@wp.func
def _fold(v: float) -> float:
    if v > 1.0:
        return 2.0 - v
    if v < -1.0:
        return -2.0 - v
    return v


@wp.func
def mandelbox_de(p: wp.vec3, scale: float, iters: int) -> wp.vec4:
    """Distance estimate to the **Mandelbox** at `p` (box-fold + sphere-fold +
    scale). ``scale`` in roughly ``[-3, 3]`` (``-1.5`` and ``2`` are classics)."""
    z = p
    dr = float(1.0)
    trap = float(1.0e10)
    min_r2 = float(0.25)
    fix_r2 = float(1.0)
    for i in range(iters):
        z = wp.vec3(_fold(z[0]), _fold(z[1]), _fold(z[2]))       # box fold
        r2 = wp.dot(z, z)
        if r2 < min_r2:                                          # sphere fold (inner)
            t = fix_r2 / min_r2
            z = z * t
            dr = dr * t
        elif r2 < fix_r2:                                        # sphere fold (outer)
            t = fix_r2 / r2
            z = z * t
            dr = dr * t
        z = z * scale + p
        dr = dr * wp.abs(scale) + 1.0
        trap = wp.min(trap, wp.length(z))
    return wp.vec4(wp.length(z) / wp.abs(dr), trap, float(iters), wp.length(z))


# --- folding fractals (IFS) -------------------------------------------------
# Reflections are distance-preserving isometries, so an iterated fold + scale is
# a valid distance estimator: DE = |z| / scale^iters. See docs/research/14.


@wp.func
def _mod2(x: float) -> float:
    """`x mod 2` (positive), without relying on a builtin mod."""
    return x - 2.0 * wp.floor(x * 0.5)


@wp.func
def _rot_xz(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] + s * p[2], p[1], -s * p[0] + c * p[2])


@wp.func
def _rot_xy(p: wp.vec3, a: float) -> wp.vec3:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec3(c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])


@wp.func
def sierpinski_de(p: wp.vec3, iters: int) -> wp.vec4:
    """Distance estimate to the **Sierpinski tetrahedron** — three plane folds
    against the tetra mirror planes, then scale ×2 about a corner vertex."""
    z = p
    trap = float(1.0e10)
    scale = float(2.0)
    for i in range(iters):
        if z[0] + z[1] < 0.0:                       # fold x+y = 0
            z = wp.vec3(-z[1], -z[0], z[2])
        if z[0] + z[2] < 0.0:                       # fold x+z = 0
            z = wp.vec3(-z[2], z[1], -z[0])
        if z[1] + z[2] < 0.0:                       # fold y+z = 0
            z = wp.vec3(z[0], -z[2], -z[1])
        z = z * scale - wp.vec3(1.0, 1.0, 1.0) * (scale - 1.0)   # scale about (1,1,1)
        trap = wp.min(trap, wp.length(z))
    de = wp.length(z) * wp.pow(scale, -float(iters))
    return wp.vec4(de, trap, float(iters), wp.length(z))


@wp.func
def _sd_box_e(p: wp.vec3, b: float) -> float:
    di = wp.vec3(wp.abs(p[0]) - b, wp.abs(p[1]) - b, wp.abs(p[2]) - b)
    mc = wp.max(di[0], wp.max(di[1], di[2]))
    q = wp.vec3(wp.max(di[0], 0.0), wp.max(di[1], 0.0), wp.max(di[2], 0.0))
    return wp.min(mc, wp.length(q))


@wp.func
def menger_de(p: wp.vec3, iters: int) -> wp.vec4:
    """**Exact** signed distance to the **Menger sponge** (Quilez): a box, then
    each level fold to the tiled cell and subtract the drilled cross."""
    d = _sd_box_e(p, 1.0)
    s = float(1.0)
    trap = float(1.0e10)
    for m in range(iters):
        ax = _mod2(p[0] * s) - 1.0
        ay = _mod2(p[1] * s) - 1.0
        az = _mod2(p[2] * s) - 1.0
        s = s * 3.0
        rx = wp.abs(1.0 - 3.0 * wp.abs(ax))
        ry = wp.abs(1.0 - 3.0 * wp.abs(ay))
        rz = wp.abs(1.0 - 3.0 * wp.abs(az))
        da = wp.max(rx, ry)
        db = wp.max(ry, rz)
        dc = wp.max(rz, rx)
        c = (wp.min(da, wp.min(db, dc)) - 1.0) / s          # the carved cross
        d = wp.max(d, c)
        trap = wp.min(trap, wp.length(wp.vec3(ax, ay, az)))
    return wp.vec4(d, trap, float(iters), d)


@wp.func
def kifs_de(p: wp.vec3, scale: float, angle: float, iters: int) -> wp.vec4:
    """Distance estimate to a **kaleidoscopic IFS** (Knighty): octant fold +
    rotation + Sierpinski folds + scale about a corner. Rotating `angle`
    reshapes the fractal architecture (temples, lattices, coral)."""
    z = p
    trap = float(1.0e10)
    off = wp.vec3(1.0, 1.0, 1.0)
    for i in range(iters):
        z = wp.vec3(wp.abs(z[0]), wp.abs(z[1]), wp.abs(z[2]))   # octant mirror
        z = _rot_xz(z, angle)                                   # kaleidoscope
        if z[0] + z[1] < 0.0:
            z = wp.vec3(-z[1], -z[0], z[2])
        if z[0] + z[2] < 0.0:
            z = wp.vec3(-z[2], z[1], -z[0])
        z = z * scale - off * (scale - 1.0)                     # scale about (1,1,1)
        z = _rot_xy(z, angle * 0.5)
        trap = wp.min(trap, wp.length(z))
    de = wp.length(z) * wp.pow(wp.abs(scale), -float(iters))
    return wp.vec4(de, trap, float(iters), wp.length(z))
