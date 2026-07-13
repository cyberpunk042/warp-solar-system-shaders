"""An immune cell — a phagocyte engulfing a pathogen.

A phagocyte (macrophage / neutrophil) hunts pathogens and **engulfs** them: it flows
its membrane out in **pseudopods** that reach around a bacterium and swallow it into a
vesicle to be digested (phagocytosis) — the cellular front line. Here a phagocyte cups
its pseudopods around a bacterium, half-engulfed. See
``docs/research/33-the-cell-up-close.md``. --frames flows the membrane.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene

_CC = wp.constant(wp.vec2(-0.25, 0.0))     # phagocyte centre
_BC = wp.constant(wp.vec2(0.62, 0.0))      # bacterium centre


@wp.kernel
def immune_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                  width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)

    col = wp.vec3(0.06, 0.04, 0.05) + wp.vec3(0.03, 0.02, 0.03) * fbm3(wp.vec3(x * 2.0, y * 2.0, 0.0), 3)

    # the bacterium (drawn first; the phagocyte wraps over it where engulfed)
    brel = wp.vec2((x - _BC[0]) / 0.18, y / 0.1)
    if wp.length(brel) < 1.0:
        col = wp.vec3(0.2, 0.6, 0.3) * (0.7 + 0.4 * hash21(wp.vec2(x * 30.0, y * 30.0)))
    col = col + wp.vec3(0.4, 0.9, 0.5) * wp.exp(-((wp.length(brel) - 1.0) / 0.06) ** 2.0) * 0.4

    # the phagocyte: irregular blob with pseudopods reaching around the bacterium
    ang = wp.atan2(y, x - _CC[0])
    rr = wp.length(p - _CC)
    lobe = fbm3(wp.vec3(wp.cos(ang) * 2.0, wp.sin(ang) * 2.0, time * 0.2), 4)
    arms = wp.exp(-((ang - 0.45) / 0.28) ** 2.0) + wp.exp(-((ang + 0.45) / 0.28) ** 2.0)
    mouth = -0.22 * wp.exp(-(ang / 0.22) ** 2.0)
    rc = 0.6 + 0.09 * lobe + 0.55 * arms + mouth
    if rr < rc:
        gran = fbm3(wp.vec3(x * 5.0, y * 5.0, 2.0), 4)
        col = wp.vec3(0.55, 0.4, 0.6) * (0.6 + 0.5 * gran)
        # lysosome granules
        g = hash21(wp.vec2(wp.floor(x * 13.0), wp.floor(y * 13.0)))
        if g > 0.92:
            fx = x * 13.0 - wp.floor(x * 13.0) - 0.5
            fy = y * 13.0 - wp.floor(y * 13.0) - 0.5
            col = col + wp.vec3(0.9, 0.6, 0.9) * wp.exp(-(fx * fx + fy * fy) * 10.0) * 0.8
        # nucleus (offset, darker kidney shape)
        dn = wp.length(p - wp.vec2(-0.45, 0.12))
        col = col * (0.6 + 0.6 * wp.smoothstep(0.18, 0.42, dn))
    # bright flowing membrane rim
    col = col + wp.vec3(0.8, 0.6, 0.9) * wp.exp(-((rr - rc) / 0.02) ** 2.0) * 0.9

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(immune_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="immune_cell",
    description="A phagocyte engulfing a bacterium — the immune cell reaching pseudopods "
                "around the pathogen, its flowing membrane cupping and swallowing it "
                "(phagocytosis), lysosome granules waiting inside. --frames flows the "
                "membrane.",
    renderer=_render,
)
