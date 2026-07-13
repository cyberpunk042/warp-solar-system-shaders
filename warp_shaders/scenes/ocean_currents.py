"""Ocean currents — the great conveyor circling the globe.

The thermohaline circulation: warm surface currents flowing poleward, cold water
returning — visualised as glowing **streamlines** advected across the oceans of a
globe, coloured warm (tropics) to cold (poles), flowing over `time`. See
``docs/research/25-earth-and-weather.md``. iMouse orbits.
"""

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def curr_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 1.0)
    if g[0] > 1.0e28 or g[0] < 0.0:
        img[i, j] = stars(rd)
        return
    n = wp.normalize(ro + rd * g[0])
    land = fbm3(n * 2.3, 5)
    is_land = wp.smoothstep(0.5, 0.57, land)

    # a streamfunction over the sphere; current lines where its phase wraps
    psi = fbm3(n * 2.6 + wp.vec3(0.0, 0.0, 1.0), 4)
    line = wp.abs(wp.sin((psi * 22.0 - time * 2.2) * 3.1416))
    current = wp.pow(line, 14.0)                        # thin moving streamlines
    lat = wp.abs(n[1])                                  # 0 equator, 1 pole
    warm = wp.vec3(1.0, 0.5, 0.25)
    cold = wp.vec3(0.3, 0.6, 1.0)
    flowcol = warm * (1.0 - lat) + cold * lat

    ocean = wp.vec3(0.02, 0.11, 0.26)
    landc = wp.vec3(0.16, 0.2, 0.12)
    surf = ocean * (1.0 - is_land) + landc * is_land
    ndl = wp.max(wp.dot(n, sun), 0.0)
    col = surf * (0.15 + 0.9 * ndl)
    col = col + flowcol * (current * (1.0 - is_land) * 1.4)   # currents on the ocean only
    rim = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
    col = col + wp.vec3(0.3, 0.5, 0.9) * (rim * (0.3 + 0.7 * ndl))
    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.4 + time * 0.05 + float(mouse[0]) * 0.01
    el = 0.3 + float(mouse[1]) * 0.01
    eye = (2.7 * math.cos(el) * math.sin(az), 2.7 * math.sin(el),
           2.7 * math.cos(el) * math.cos(az))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=44.0, aspect=width / height)
    saz = az + 0.9
    sun = wp.vec3(math.cos(0.5) * math.sin(saz), math.sin(0.5), math.cos(0.5) * math.cos(saz))
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(curr_kernel, dim=(height, width),
              inputs=[img, cam, sun, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=1.3, strength=0.35,
                     radius=max(2, int(width * 0.01)), passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="ocean_currents",
    description="Ocean currents — the thermohaline 'great conveyor' as glowing "
                "streamlines flowing across a globe's oceans, warm (tropics) to cold "
                "(poles). iMouse orbits; --frames flows the currents.",
    renderer=_render,
)
