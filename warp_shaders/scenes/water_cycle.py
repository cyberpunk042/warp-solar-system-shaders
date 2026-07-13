"""The water cycle — evaporation, condensation, precipitation.

Over a coastline: the Sun **evaporates** ocean water, the vapour rises and
**condenses** into clouds, which **precipitate** rain back to the sea — the loop
that moves water around the planet. Loops with `time`. See
``docs/research/25-earth-and-weather.md``. --frames animates the cycle.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def cycle_kernel(img: wp.array2d(dtype=wp.vec3), time: float, aspect: float,
                 width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0
    sea_level = -0.35

    # sky gradient + warm sun upper-left
    col = wp.vec3(0.28, 0.5, 0.82) * (0.5 + 0.5 * y) + wp.vec3(0.75, 0.82, 0.9) * (0.5 - 0.5 * y)
    sun = wp.exp(-((x + 1.1) * (x + 1.1) + (y - 0.72) * (y - 0.72)) * 2.0)
    col = col + wp.vec3(1.0, 0.9, 0.6) * (sun * 0.9)

    # rising vapour columns (evaporation) — a few plumes lifting off the sea
    vap = fbm3(wp.vec3(x * 2.4, y * 1.6 - time * 0.6, 7.0), 5)
    vapour = wp.smoothstep(0.5, 0.78, vap) * wp.smoothstep(0.55, -0.2, y) * wp.smoothstep(sea_level, 0.0, y)
    col = col + wp.vec3(0.85, 0.9, 0.95) * (vapour * 0.7)

    # clouds (condensation) in the upper sky
    cl = fbm3(wp.vec3(x * 1.3 + time * 0.06, y * 2.4, 3.0), 5)
    cloudmask = wp.smoothstep(0.46, 0.7, cl) * wp.smoothstep(0.15, 0.5, y)
    col = col * (1.0 - cloudmask) + wp.vec3(0.97, 0.98, 1.0) * cloudmask
    col = col * (1.0 - cloudmask * 0.3) + wp.vec3(0.55, 0.6, 0.72) * (cloudmask * (1.0 - cl) * 0.4)

    # rain (precipitation) — diagonal streaks below the clouds
    rain = fbm3(wp.vec3(x * 18.0 + y * 10.0, y * 5.0 - time * 9.0, 2.0), 2)
    band = wp.smoothstep(0.2, 0.05, y) * wp.smoothstep(-0.4, 0.0, y)     # between cloud and sea
    rainmask = wp.smoothstep(0.6, 0.85, rain) * band
    col = col + wp.vec3(0.6, 0.72, 0.9) * (rainmask * 0.6)

    # the sea (horizontal receding ripples + a bright horizon glint)
    if y < sea_level:
        depth = sea_level - y
        ripple = 0.5 + 0.5 * wp.sin(depth * 45.0 - time * 2.0
                                    + fbm3(wp.vec3(x * 3.0, depth * 4.0, 0.0), 3) * 4.0)
        sea = wp.vec3(0.02, 0.16, 0.3) + wp.vec3(0.04, 0.09, 0.12) * ripple * wp.exp(-depth * 1.6)
        glint = wp.pow(ripple, 10.0) * wp.exp(-depth * 3.0)
        sea = sea + wp.vec3(1.0, 0.92, 0.7) * (glint * 0.5)
        col = sea
    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(cycle_kernel, dim=(height, width),
              inputs=[img, float(time), float(width / height), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=1.3, strength=0.25,
                     radius=max(2, int(width * 0.01)), passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="water_cycle",
    description="The water cycle — the Sun evaporating the sea, vapour rising and "
                "condensing into clouds, and rain falling back to the ocean over a "
                "coastline. --frames animates the cycle.",
    renderer=_render,
)
