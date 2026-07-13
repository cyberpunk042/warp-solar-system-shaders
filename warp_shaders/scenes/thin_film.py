"""Thin-film interference — the colours of a soap bubble.

A soap film reflects light off both its surfaces; the two reflections interfere, and
because the condition depends on **film thickness** and **viewing angle** and
**wavelength**, you see swirling iridescent bands. The film drains thinner at the top
(a black spot where it is about to pop). Reflectance per wavelength ≈ sin²(π·OPD/λ)
with OPD = 2·n·d·cosθ. See ``docs/research/30-light-and-optics.md``. --frames swirls
the film.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.kernel
def film_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    g = _rs(ro, rd, 1.2)
    if g[0] < 0.0 or g[0] > 1.0e28:
        # soft dark studio background
        bg = wp.vec3(0.03, 0.035, 0.05) * (0.6 + 0.6 * v) \
            + wp.vec3(0.06, 0.06, 0.08) * wp.exp(-((u - 0.4) ** 2.0 + (v - 0.4) ** 2.0) * 1.5)
        img[i, j] = bg
        return
    p = ro + rd * g[0]
    n = wp.normalize(p)
    cosv = wp.max(wp.dot(n, -rd), 0.0)

    # film thickness (nm): swirling, draining thinner toward the top
    swirl = fbm3(n * 3.6 + wp.vec3(0.0, -time * 0.18, time * 0.05), 5)
    grav = wp.smoothstep(-0.3, 1.0, n[1])
    d = 560.0 + 360.0 * swirl - 340.0 * grav
    d = wp.max(d, 8.0)

    opd = 2.0 * 1.33 * d * cosv
    rr = 0.5 - 0.5 * wp.cos(6.2831 * opd / 650.0)
    gg = 0.5 - 0.5 * wp.cos(6.2831 * opd / 550.0)
    bb = 0.5 - 0.5 * wp.cos(6.2831 * opd / 450.0)
    film = wp.vec3(rr, gg, bb)

    fres = wp.pow(1.0 - cosv, 3.0)                      # brighter, whiter at the rim
    col = film * (0.5 + 1.0 * fres) + wp.vec3(1.0, 1.0, 1.0) * fres * 0.3
    col = col * wp.smoothstep(8.0, 55.0, d)            # black drain-spot at the top
    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=3.6, fov=40.0, el0=0.1, auto=0.08)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(film_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="thin_film",
    description="A soap bubble's iridescence — thin-film interference: the two surface "
                "reflections add or cancel per wavelength depending on film thickness and "
                "viewing angle, drawing swirling colour bands that drain thinner (to a "
                "black spot) at the top. --frames swirls the film.",
    renderer=_render,
)
