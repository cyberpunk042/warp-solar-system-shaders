"""A hurricane — the satellite view of a tropical cyclone.

Log-spiral **rainbands** wheeling counter-clockwise around a clear, calm **eye**,
ringed by the bright towering **eyewall**, over warm ocean. Spins with `time`. See
``docs/research/25-earth-and-weather.md``. --frames animates the rotation.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def hurricane_kernel(img: wp.array2d(dtype=wp.vec3), rot: float, aspect: float,
                     time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0
    r = wp.sqrt(x * x + y * y)
    th = wp.atan2(y, x)

    eye_r = 0.13
    # log-spiral bands, wheeling in time
    spiral = th * 3.0 + wp.log(r + 0.05) * 6.5 - rot
    band = 0.5 + 0.5 * wp.sin(spiral)
    detail = 0.5 + 0.6 * fbm3(wp.vec3(x * 3.5, y * 3.5, time * 0.15), 5)
    # clouds fade out inside the eye and past the storm edge
    inside_eye = wp.smoothstep(eye_r * 0.8, eye_r * 1.35, r)
    edge = 1.0 - wp.smoothstep(0.85, 1.15, r)
    cloud = wp.clamp((band * 0.6 + detail * 0.7 - 0.55) * 3.0, 0.0, 1.0) * inside_eye * edge
    # bright eyewall ring
    eyewall = wp.exp(-((r - eye_r) / 0.035) * ((r - eye_r) / 0.035)) * edge

    ocean = wp.vec3(0.03, 0.12, 0.25) * (0.7 + 0.4 * detail)
    cloudc = wp.vec3(0.92, 0.94, 0.98)
    col = ocean * (1.0 - cloud) + cloudc * cloud
    col = col + wp.vec3(1.0, 1.0, 1.0) * (eyewall * 0.7)
    # shade the eye interior a touch warmer (calm, sunlit sea)
    col = col + wp.vec3(0.15, 0.2, 0.28) * (1.0 - inside_eye) * edge
    img[i, j] = col


def _render(width, height, time, mouse, device):
    rot = time * 1.3
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(hurricane_kernel, dim=(height, width),
              inputs=[img, float(rot), float(width / height), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.2, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="hurricane",
    description="A hurricane from orbit — log-spiral rainbands wheeling around a "
                "clear eye ringed by a bright eyewall, over warm ocean. --frames "
                "animates the rotation.",
    renderer=_render,
)
