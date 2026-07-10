"""Neutron — three quarks (up, down, down) bound by gluon flux tubes.

Same ``nucleon`` primitive as the proton, with the udd flavor content and a
cool (charge-neutral) confinement tint. iMouse orbits the camera.
"""

import warp as wp

from ..particles import camera_ray, nucleon, orbit_ro
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

    ro = orbit_ro(time, mouse, res, 4.0)
    rd = camera_ray(uv, ro, wp.vec3(0.0, 0.0, 0.0), 1.6)

    col = nucleon(ro, rd, time, 0)
    col = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                  wp.pow(wp.max(col[1], 0.0), 0.4545),
                  wp.pow(wp.max(col[2], 0.0), 0.4545))
    img[i, j] = col


SCENE = Scene(
    name="neutron",
    kernel=render_kernel,
    description="Neutron: up+down+down quarks bound by gluon flux tubes (color-neutral). iMouse orbits.",
)
