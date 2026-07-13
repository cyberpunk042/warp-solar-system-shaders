"""A Matrioshka brain — a star turned into a mind.

The ultimate Type II computer: nested Dyson shells, each running on the **waste heat**
of the shell inside it, extracting nearly all a star's energy as computation. The
shells glow cooler outward — hot blue-white at the core to deep infrared red at the
rim — a star's whole output becoming thought. See
``docs/research/29-megastructures-and-far-future.md``. --frames rotates the shells.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.hash import hash21
from ..subatomic.field import void
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _shell(p: wp.vec3, R: float, col: wp.vec3, spin: float) -> wp.vec3:
    r = wp.length(p)
    glow = wp.exp(-((r - R) / 0.028) * ((r - R) / 0.028))
    if glow < 0.001:
        return wp.vec3(0.0, 0.0, 0.0)
    n = p / r
    theta = wp.acos(wp.clamp(n[1], -1.0, 1.0))
    phi = wp.atan2(n[2], n[0]) + spin
    ct = wp.floor(theta / 3.14159 * 22.0)
    cp = wp.floor((phi / 6.2831 + 0.5) * 44.0)
    present = hash21(wp.vec2(ct + R * 10.0, cp * 1.3))
    ft = theta / 3.14159 * 22.0 - ct - 0.5
    fp = (phi / 6.2831 + 0.5) * 44.0 - cp - 0.5
    border = wp.max(wp.abs(ft), wp.abs(fp))
    panel = float(0.0)
    if present > 0.3:
        panel = 0.5 + 0.5 * wp.smoothstep(0.42, 0.34, border)     # panel + hot seam
    return col * (glow * panel)


@wp.kernel
def brain_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, spin: float,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 1.75)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 90.0
    t = t0 + dt * 0.5
    acc = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(90):
        p = ro + rd * t
        r = wp.length(p)
        # central star
        acc = acc + wp.vec3(0.7, 0.85, 1.0) * wp.exp(-(r / 0.14) * (r / 0.14)) * 3.0 * dt
        # nested shells, cooler (redder) outward
        acc = acc + _shell(p, 0.55, wp.vec3(0.8, 0.9, 1.0), spin * 1.4) * dt * 2.2
        acc = acc + _shell(p, 0.9, wp.vec3(1.0, 0.85, 0.5), spin * 1.0) * dt * 2.0
        acc = acc + _shell(p, 1.25, wp.vec3(1.0, 0.45, 0.18), spin * 0.7) * dt * 1.8
        acc = acc + _shell(p, 1.6, wp.vec3(0.7, 0.16, 0.07), spin * 0.5) * dt * 1.6
        t += dt
    img[i, j] = acc + void(rd) * 0.6


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=4.6, fov=42.0, el0=0.2, auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(brain_kernel, dim=(height, width),
              inputs=[img, cam, float(time * 0.1), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.6, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="matrioshka_brain",
    description="A Matrioshka brain — nested Dyson shells around a star, each running on "
                "the waste heat of the shell within, glowing cooler outward (blue-white "
                "core to deep infrared rim): a star's whole output turned into "
                "computation. --frames rotates the shells.",
    renderer=_render,
)
