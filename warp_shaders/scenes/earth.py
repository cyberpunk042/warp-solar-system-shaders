"""Earth from space — a realistic ray-sphere globe with atmosphere.

Fully procedural (no texture asset): domain-warped continents, oceans with a
specular sun-glint, drifting clouds, a day/night terminator with night-side city
lights, and an atmospheric-scattering blue rim + warm sunset limb over a
starfield. Shading lives in ``warp_shaders/earthgfx.py`` (shared with the Earth
blast simulation). iMouse orbits the camera.

For photoreal geography, swap the procedural landmask in ``earthgfx.surface``
for a sampled NASA Blue-Marble equirectangular texture; everything else stays.
"""

import warp as wp

from ..earthgfx import earth_color
from ..particles import camera_ray, orbit_ro
from ..scene import Scene

_SUN = wp.constant(wp.vec3(-0.5, 0.28, 0.82))   # ~90 deg off the camera arc -> terminator


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

    ro = orbit_ro(time, mouse, res, 3.0)
    rd = camera_ray(wp.vec2(uvx, uvy), ro, wp.vec3(0.0, 0.0, 0.0), 1.7)

    col = earth_color(ro, rd, wp.normalize(_SUN), time, 0.0)
    col = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                  wp.pow(wp.max(col[1], 0.0), 0.4545),
                  wp.pow(wp.max(col[2], 0.0), 0.4545))
    img[i, j] = col


SCENE = Scene(
    name="earth",
    kernel=render_kernel,
    description="Realistic Earth from space: atmosphere, oceans, clouds, city lights, terminator. iMouse orbits.",
)
