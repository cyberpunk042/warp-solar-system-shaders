"""Electron — a point lepton rendered as a 1s probability cloud.

An electron is point-like, but in an atom it exists as a probability
distribution. Here it's the hydrogen 1s orbital density ~ exp(-r/a), integrated
volumetrically along the ray, with quantum 'sparkle' from 3D noise.
"""

import warp as wp

from ..particles import camera_ray, noise3, orbit_ro, ray_sphere
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

    ro = orbit_ro(time, mouse, res, 5.0)
    rd = camera_ray(uv, ro, wp.vec3(0.0, 0.0, 0.0), 1.6)

    center = wp.vec3(0.0, 0.0, 0.0)
    a = 0.5             # orbital decay scale
    cloud_r = 3.0
    t0, t1, hit = ray_sphere(ro, rd, center, cloud_r)

    col = wp.vec3(0.0, 0.0, 0.0)
    if hit == 1:
        t0 = wp.max(t0, 0.0)
        n = 56
        dt = (t1 - t0) / float(n)
        acc = float(0.0)
        for s in range(56):
            tt = t0 + (float(s) + 0.5) * dt
            p = ro + rd * tt
            r = wp.length(p - center)
            dens = wp.exp(-r / a)
            dens = dens * dens * dens  # concentrate toward the nucleus; edges fall to black
            spark = 0.25 + 0.9 * noise3(p * 4.5 + wp.vec3(0.0, 0.0, time * 1.5))
            acc += dens * spark * dt
        acc = wp.max(acc - 0.01, 0.0)  # clip the low-level haze to black
        col = wp.vec3(0.45, 0.65, 1.0) * (acc * 24.0)

    col = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                  wp.pow(wp.max(col[1], 0.0), 0.4545),
                  wp.pow(wp.max(col[2], 0.0), 0.4545))
    img[i, j] = col


SCENE = Scene(
    name="electron",
    kernel=render_kernel,
    description="Electron as a hydrogen 1s probability cloud (exp(-r/a)) with quantum sparkle. iMouse orbits.",
)
