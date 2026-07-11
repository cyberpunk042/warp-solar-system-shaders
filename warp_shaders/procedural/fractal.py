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
