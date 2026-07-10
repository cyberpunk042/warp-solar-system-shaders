"""Quark — a single color-charged quark.

A free quark can't be isolated (confinement), so it's shown as one glowing orb
whose QCD color charge cycles red->green->blue over time, with a plasma-textured
surface and gluon wisps radiating outward (the field that would bind it).
"""

import warp as wp

from ..particles import camera_ray, emitter, noise3, orbit_ro
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

    ro = orbit_ro(time, mouse, res, 3.2)
    rd = camera_ray(uv, ro, wp.vec3(0.0, 0.0, 0.0), 1.6)

    # Color charge cycling through r/g/b.
    qcol = wp.vec3(0.5 + 0.5 * wp.sin(time),
                   0.5 + 0.5 * wp.sin(time + 2.094),
                   0.5 + 0.5 * wp.sin(time + 4.188))

    core = emitter(ro, rd, wp.vec3(0.0, 0.0, 0.0), 0.55)
    # Surface plasma: modulate by 3D noise sampled along the view direction.
    tex = 0.6 + 0.4 * noise3(rd * 6.0 + wp.vec3(0.0, 0.0, time))

    col = qcol * (core * tex)
    col = col + wp.vec3(1.0, 1.0, 1.0) * (emitter(ro, rd, wp.vec3(0.0, 0.0, 0.0), 0.12) * 0.6)

    # Gluon wisps radiating to a few nearby points.
    for k in range(4):
        a = float(k) * 1.571 + time * 0.5
        tip = wp.vec3(1.2 * wp.cos(a), 1.2 * wp.sin(a), 0.4 * wp.sin(time + float(k)))
        col = col + qcol * (emitter(ro, rd, tip * 0.6, 0.07) * 0.25)

    col = wp.vec3(wp.pow(wp.max(col[0], 0.0), 0.4545),
                  wp.pow(wp.max(col[1], 0.0), 0.4545),
                  wp.pow(wp.max(col[2], 0.0), 0.4545))
    img[i, j] = col


SCENE = Scene(
    name="quark",
    kernel=render_kernel,
    description="A single quark: color charge cycling r->g->b, plasma surface, gluon wisps. iMouse orbits.",
)
