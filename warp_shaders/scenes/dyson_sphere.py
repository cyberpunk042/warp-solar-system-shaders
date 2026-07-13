"""A Dyson sphere — a star wrapped to capture its whole output.

A **Type II** civilisation surrounds its star with a swarm of collector panels to
capture most of its light (Dyson, 1960). From outside the star dims in visible light
while the structure re-radiates the absorbed energy as **warm waste heat** — its
telltale infrared signature. Here a partial shell of panels cages a brilliant star,
its light blazing through the gaps. See ``docs/research/29-megastructures-and-far-future.md``.
--frames rotates the swarm.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..subatomic.field import void
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _star(p: wp.vec3, time: float) -> wp.vec3:
    n = wp.normalize(p)
    gran = fbm3(n * 6.0 + wp.vec3(0.0, time * 0.2, 0.0), 4)
    spot = fbm3(n * 2.5 - wp.vec3(time * 0.1, 0.0, 0.0), 3)
    c = wp.vec3(1.0, 0.82, 0.45) * (1.6 + 0.9 * gran) - wp.vec3(0.3, 0.3, 0.3) * spot
    return c * 1.6


@wp.kernel
def dyson_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, spin: float,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    col = void(rd)
    gshell = _rs(ro, rd, 1.5)
    gstar = _rs(ro, rd, 0.82)

    panel_hit = int(0)
    if gshell[0] > 0.0 and gshell[0] < 1.0e29:
        p = ro + rd * gshell[0]
        n = wp.normalize(p)
        theta = wp.acos(wp.clamp(n[1], -1.0, 1.0))
        phi = wp.atan2(n[2], n[0]) + spin
        ct = wp.floor(theta / 3.14159 * 15.0)
        cp = wp.floor((phi / 6.2831 + 0.5) * 30.0)
        present = hash21(wp.vec2(ct, cp * 1.3))
        # cell-local coords for the panel bevel / gap
        ft = theta / 3.14159 * 15.0 - ct - 0.5
        fp = (phi / 6.2831 + 0.5) * 30.0 - cp - 0.5
        border = wp.max(wp.abs(ft), wp.abs(fp))
        if present > 0.44 and border < 0.44:
            var = 0.5 + 0.5 * hash21(wp.vec2(ct * 2.1, cp))
            warm = wp.vec3(1.0, 0.4, 0.12) * (0.12 + 0.22 * var)   # dim IR re-radiation
            metal = wp.vec3(0.03, 0.035, 0.05)
            frame = wp.smoothstep(0.44, 0.37, border)              # glowing seams
            col = metal + warm + wp.vec3(0.6, 0.75, 0.95) * (1.0 - frame) * 0.5
            panel_hit = 1

    if panel_hit == 0 and gstar[0] > 0.0 and gstar[0] < 1.0e29:
        col = _star(ro + rd * gstar[0], time)
    elif panel_hit == 0 and gshell[0] > 0.0 and gshell[0] < 1.0e29:
        # gap in the shell but star missed: warm inner glow bleeding out
        col = col + wp.vec3(1.0, 0.55, 0.2) * 0.25

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=4.4, fov=40.0, el0=0.18,
                       auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(dyson_kernel, dim=(height, width),
              inputs=[img, cam, float(time * 0.15), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="dyson_sphere",
    description="A Dyson sphere — a brilliant star caged in a partial swarm of collector "
                "panels re-radiating warm infrared waste heat, its light blazing through "
                "the gaps (a Type II civilisation capturing a star's whole output). "
                "--frames rotates the swarm.",
    renderer=_render,
)
