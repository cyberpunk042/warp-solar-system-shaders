"""A rainbow — sunlight bent back by raindrops.

Each raindrop refracts sunlight in, reflects it off its back, and refracts it out,
bending it ~**42°** from the antisolar point — with each colour at its own angle (red
outer, violet inner). A fainter **secondary** bow at ~51° has its colours reversed,
and the sky between them (**Alexander's band**) is darker. See
``docs/research/30-light-and-optics.md``. --frames drifts the rain.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import wavelength_rgb
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def rainbow_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                   width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0

    horizon = -0.32
    # sky gradient, brighter toward the horizon
    sky = wp.vec3(0.14, 0.26, 0.42) * (0.7 + 0.5 * (1.0 - y)) \
        + wp.vec3(0.42, 0.47, 0.52) * wp.smoothstep(0.4, horizon, y) * 0.4
    col = sky

    # bows centred on the antisolar point (below the horizon)
    asp = wp.vec2(0.05, -0.85)
    r = wp.length(wp.vec2(x - asp[0], y - asp[1]))
    if y > horizon:
        # rain veil brightens where the bows are
        rain = 0.5 + 0.5 * fbm3(wp.vec3(x * 3.0, y * 3.0 - time * 0.5, 0.0), 3)
        # primary bow ~42°: violet inner .. red outer
        s1 = (r - 1.02) / (1.18 - 1.02)
        if s1 > 0.0 and s1 < 1.0:
            env = wp.smoothstep(0.0, 0.12, s1) * wp.smoothstep(1.0, 0.88, s1)
            col = col + wavelength_rgb(400.0 + s1 * (645.0 - 400.0)) * env * rain * 1.7
        # Alexander's band (darker sky between the bows)
        if r > 1.18 and r < 1.30:
            col = col * 0.78
        # secondary bow ~51°: colours reversed (red inner .. violet outer)
        s2 = (r - 1.30) / (1.45 - 1.30)
        if s2 > 0.0 and s2 < 1.0:
            env = wp.smoothstep(0.0, 0.18, s2) * wp.smoothstep(1.0, 0.82, s2)
            col = col + wavelength_rgb(645.0 - s2 * (645.0 - 400.0)) * env * rain * 0.8
        # faint falling rain streaks
        streak = hash21(wp.vec2(wp.floor(x * 90.0), wp.floor((y + time * 1.2) * 12.0)))
        if streak > 0.985:
            col = col + wp.vec3(0.5, 0.55, 0.6) * 0.25
    else:
        # dark-green wet landscape
        hill = 0.5 + 0.5 * fbm3(wp.vec3(x * 2.0, 0.0, 5.0), 4)
        col = wp.vec3(0.05, 0.12, 0.06) * (0.5 + 0.7 * hill)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(rainbow_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.2, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="rainbow",
    description="A rainbow over a rainy landscape — the primary bow at ~42° (red outer, "
                "violet inner), a fainter reversed secondary at ~51°, and the darker "
                "Alexander's band between them. --frames drifts the rain.",
    renderer=_render,
)
