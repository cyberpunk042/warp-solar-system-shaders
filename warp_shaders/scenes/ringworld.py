"""A ringworld — a habitable band spun around a star.

Larry Niven's **Ringworld** (1970): a band ~1 AU in radius spun around its star, its
inner surface a habitable strip millions of km wide, spin providing gravity. An inner
ring of **shadow squares** makes day and night. Seen at a shallow angle the near rim
sweeps across the foreground and the far arc rises past the star. See
``docs/research/29-megastructures-and-far-future.md``. --frames spins it.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.field import void
from ..subatomic.render import orbit_camera
from ..scene import Scene

_R = 3.0
_W = 0.42


@wp.kernel
def ring_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, spin: float,
                time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = void(rd)
    # the central star
    gs = _rs(ro, rd, 0.6)
    tstar = float(1e30)
    if gs[0] > 0.0 and gs[0] < 1.0e29:
        tstar = gs[0]

    # ring: intersect the ring plane (y=0), keep the annulus R-W..R+W
    tring = float(1e30)
    rcol = wp.vec3(0.0, 0.0, 0.0)
    if wp.abs(rd[1]) > 1e-5:
        t = -ro[1] / rd[1]
        if t > 0.0:
            p = ro + rd * t
            rr = wp.length(wp.vec2(p[0], p[2]))
            if rr > _R - _W and rr < _R + _W:
                tring = t
                a = wp.atan2(p[2], p[0]) + spin
                f = (rr - (_R - _W)) / (2.0 * _W)          # 0 inner edge .. 1 outer
                # habitable strip: ocean/land bands, lit by the star (inner-facing)
                land = fbm3(wp.vec3(a * 6.0, f * 4.0, 0.0), 4)
                ocean = wp.vec3(0.05, 0.2, 0.4)
                green = wp.vec3(0.2, 0.5, 0.2)
                surf = ocean * (1.0 - wp.smoothstep(0.45, 0.6, land)) \
                    + green * wp.smoothstep(0.45, 0.6, land)
                # shadow squares -> alternating day/night sectors
                night = wp.smoothstep(0.0, 0.3, wp.sin(a * 14.0) - 0.2)
                day = 1.0 - night
                edge = wp.smoothstep(0.0, 0.08, f) * wp.smoothstep(1.0, 0.92, f)
                lights = wp.vec3(1.0, 0.85, 0.5) * night * (0.3 + 0.5 * land) * 0.4
                rcol = (surf * (0.25 + 1.0 * day) + lights) * edge
                rcol = rcol + wp.vec3(0.5, 0.7, 1.0) * (1.0 - edge) * 0.2   # rim walls

    if tring < tstar and tring < 1e29:
        col = rcol
    elif tstar < 1e29:
        n = wp.normalize(ro + rd * tstar)
        gran = fbm3(n * 6.0 + wp.vec3(0.0, time * 0.2, 0.0), 4)
        col = wp.vec3(1.0, 0.9, 0.6) * (1.7 + 0.8 * gran)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=7.5, fov=40.0, el0=0.1,
                       auto=0.08)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(ring_kernel, dim=(height, width),
              inputs=[img, cam, float(time * 0.05), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.011))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="ringworld",
    description="A ringworld — a habitable band spun around a star, its inner surface a "
                "strip of ocean and land lit by the star, an inner ring of shadow squares "
                "making day and night, the far arc rising past the star. --frames spins it.",
    renderer=_render,
)
