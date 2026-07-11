"""Reusable shading helpers — small, map-independent device functions.

Warp has no function pointers, so a raymarcher must inline its scene `map()`.
These helpers are the *map-independent* pieces every scene tends to re-derive:
distance fog / aerial perspective, a sun disk + glow for the background, and a
two-stop sky gradient. Keeping them here means a new world composes the look
instead of copy-pasting it.
"""

import warp as wp

_PI = 3.14159265


@wp.func
def apply_fog(col: wp.vec3, dist: float, fog_col: wp.vec3, density: float) -> wp.vec3:
    """Blend `col` toward `fog_col` with exponential distance fog.

    `f = 1 - exp(-density * dist)`; `density` is per-unit-distance extinction, so
    larger = thicker haze. Cheap aerial perspective for terrain/heightfield scenes.
    """
    f = 1.0 - wp.exp(-wp.max(density, 0.0) * wp.max(dist, 0.0))
    return col * (1.0 - f) + fog_col * f


@wp.func
def sun_disk(rd: wp.vec3, sun: wp.vec3, disk_col: wp.vec3,
             size: float, glow: float) -> wp.vec3:
    """A sun disk + soft glow for a background/sky ray.

    `size` in ~[0.9990, 0.9999] sets the disk edge (larger = tighter); `glow`
    scales a wide `pow(max(dot,0), 8)` halo around it. Returns additive radiance.
    """
    mu = wp.dot(rd, sun)
    disk = wp.smoothstep(size, size + 0.0004, mu)
    halo = wp.pow(wp.max(mu, 0.0), 8.0) * glow
    return disk_col * (disk * 12.0 + halo)


@wp.func
def sky_gradient(rd: wp.vec3, horizon: wp.vec3, zenith: wp.vec3) -> wp.vec3:
    """Two-stop vertical sky gradient by ray elevation (`rd[1]` in [-1,1])."""
    t = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    return horizon * (1.0 - t) + zenith * t
