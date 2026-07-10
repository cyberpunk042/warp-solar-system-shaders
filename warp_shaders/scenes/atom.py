"""Hydrogen atom — the top of the bottom-up build.

A proton nucleus (tiny, bright) at the center, wrapped by the electron's 1s
probability cloud (large, diffuse). This composes the same primitives as the
proton and electron scenes.

NOT to scale: a real nucleus is ~1e-5 of the atom. The nucleus is exaggerated so
its quark structure is faintly visible inside the cloud.
"""

import warp as wp

from ..particles import camera_ray, noise3, nucleon, orbit_ro, ray_sphere
from ..scene import Scene


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))
    uvx = ((float(j) + 0.5) - 0.5 * res[0]) / res[1]
    uvy = ((float(height - 1 - i) + 0.5) - 0.5 * res[1]) / res[1]
    uv = wp.vec2(uvx, uvy)

    ro = orbit_ro(time, mouse, res, 6.5)
    rd = camera_ray(uv, ro, wp.vec3(0.0, 0.0, 0.0), 1.6)

    center = wp.vec3(0.0, 0.0, 0.0)

    # Electron 1s cloud (large, faint), integrated volumetrically.
    a = 0.85
    cloud_r = 3.5
    t0, t1, hit = ray_sphere(ro, rd, center, cloud_r)
    col = wp.vec3(0.0, 0.0, 0.0)
    if hit == 1:
        t0 = wp.max(t0, 0.0)
        n = 48
        dt = (t1 - t0) / float(n)
        acc = float(0.0)
        for s in range(48):
            tt = t0 + (float(s) + 0.5) * dt
            p = ro + rd * tt
            r = wp.length(p - center)
            dens = wp.exp(-r / a)
            dens = dens * dens * dens
            spark = 0.3 + 0.7 * noise3(p * 3.5 + wp.vec3(0.0, 0.0, time))
            acc += dens * spark * dt
        acc = wp.max(acc - 0.008, 0.0)  # clip the low-level haze to black
        col = wp.vec3(0.35, 0.55, 1.0) * (acc * 16.0)

    # Nucleus: a small bright proton at the center (scale the local space down).
    scale = 6.0
    col = col + nucleon(ro * scale, rd, time, 1) * 1.6

    col = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                  wp.pow(wp.max(col[1], 0.0), 0.4545),
                  wp.pow(wp.max(col[2], 0.0), 0.4545))
    img[i, j] = col


SCENE = Scene(
    name="atom",
    kernel=render_kernel,
    description="Hydrogen: a proton nucleus inside the electron's 1s cloud (not to scale). iMouse orbits.",
)
