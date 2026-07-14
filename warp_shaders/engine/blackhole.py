"""Shared relativistic black-hole rendering helpers (device ``@wp.func``).

The physics common to the geodesic-traced black-hole scenes (``gargantua``, ``kerr``,
``binary_bh``, ``wormhole``): a relativistic **accretion-disk emission** model and the
**cosmic background**. The per-scene metric (Schwarzschild / Kerr / two-body / Ellis) lives in
each scene's integrator; this module is what a ray *sees* when it hits the disk or escapes.

Units are geometric with the Schwarzschild radius ``r_s = 1`` (so ``GM = 0.5``).
"""

import warp as wp

from .color import kelvin_to_rgb
from .sky import milky_way, starfield


@wp.func
def disk_emission(cp: wp.vec3, pdir: wp.vec3, time: float,
                  r_in: float, r_out: float, temp0: float, bright: float) -> wp.vec3:
    """Emission of a thin equatorial accretion disk where a ray crosses ``y = 0``.

    ``cp`` crossing point, ``pdir`` the (unit) photon direction. Combines a Shakura–Sunyaev
    ``T ∝ r^-3/4`` blackbody gradient, relativistic Doppler beaming (approaching side brighter +
    bluer, ``∝ D³``), gravitational redshift, and turbulent Keplerian banding. Returns black
    outside ``[r_in, r_out]``."""
    rc = wp.sqrt(cp[0] * cp[0] + cp[2] * cp[2])
    if rc < r_in or rc > r_out:
        return wp.vec3(0.0, 0.0, 0.0)

    temp = temp0 * wp.pow(r_in / rc, 0.6)
    beta = wp.min(wp.sqrt(0.5 / rc), 0.62)                          # Keplerian speed (GM = 0.5)
    cph = cp / wp.length(cp)
    tang = wp.normalize(wp.cross(wp.vec3(0.0, 1.0, 0.0), cph))      # prograde orbital direction
    approach = wp.dot(tang, -pdir)
    dopp = 1.0 / wp.max(1.0 - beta * approach, 0.2)
    grav = wp.sqrt(wp.max(1.0 - 1.0 / rc, 0.02))

    col = kelvin_to_rgb(wp.clamp(temp * dopp * grav, 1200.0, 40000.0))

    ang = wp.atan2(cp[2], cp[0]) + time * (2.4 / wp.pow(rc, 1.5))   # inner rings orbit faster
    tex = 0.62 + 0.30 * wp.sin(ang * 5.0 + rc * 2.6) + 0.16 * wp.sin(ang * 13.0 - rc * 1.7)
    fall = wp.pow(r_in / rc, 1.3)
    edge = 1.0 + 1.8 * wp.exp(-(rc - r_in) * 1.3)                   # bright inner rim
    beam = wp.min(dopp * dopp * dopp, 6.0)
    return col * (fall * edge * wp.max(tex, 0.0) * beam * grav * bright)


@wp.func
def cosmic_background(rd: wp.vec3, mw: float) -> wp.vec3:
    """The deep-sky background a ray sees on escape: procedural stars plus an optional faint
    Milky-Way band (``mw`` intensity; 0 = plain starfield). Lensed automatically because ``rd``
    is the ray's *bent* final direction."""
    col = starfield(rd)
    if mw > 0.0:
        col = col + milky_way(rd, wp.vec3(0.2, 1.0, 0.35), mw)
    return col
