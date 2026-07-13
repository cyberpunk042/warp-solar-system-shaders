"""An interferometer — interference fringes from split light.

Coherent light split into two paths and recombined interferes: where the path
difference is a whole wavelength the beams add (bright), half a wavelength they cancel
(dark). A **Michelson** interferometer maps that path difference to concentric fringes
— coloured near zero path difference (white-light fringes), washing to grey further
out — that shift as a mirror moves. See ``docs/research/30-light-and-optics.md``.
--frames scans the mirror.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import wavelength_rgb
from ..scene import Scene


@wp.kernel
def interf_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, d0: float,
                  width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    r2 = x * x + y * y

    if r2 > 1.0:                                    # outside the aperture: dark optics
        img[i, j] = wp.vec3(0.03, 0.032, 0.04) * (0.6 + 0.4 * y)
        return

    # optical path difference: a scanned offset + curvature (equal-inclination fringes)
    opd = d0 + 13000.0 * r2                         # nanometres
    acc = wp.vec3(0.0, 0.0, 0.0)
    norm = wp.vec3(0.0, 0.0, 0.0)
    for k in range(7):
        lam = 430.0 + float(k) * (660.0 - 430.0) / 6.0
        inten = 0.5 + 0.5 * wp.cos(6.2831 * opd / lam)
        crgb = wavelength_rgb(lam)
        acc = acc + crgb * inten
        norm = norm + crgb
    col = wp.vec3(acc[0] / norm[0], acc[1] / norm[1], acc[2] / norm[2])
    col = (col - wp.vec3(0.5, 0.5, 0.5)) * 1.7 + wp.vec3(0.42, 0.42, 0.42)  # contrast
    col = wp.vec3(wp.max(col[0], 0.0), wp.max(col[1], 0.0), wp.max(col[2], 0.0))
    col = col * wp.smoothstep(1.0, 0.9, r2)          # soft aperture edge
    img[i, j] = col


def _render(width, height, time, mouse, device):
    d0 = float(time * 900.0)                          # mirror scan (nm)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(interf_kernel, dim=(height, width),
              inputs=[img, float(width / height), d0, int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.006))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.2, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="interferometer",
    description="A Michelson interferometer — concentric interference fringes from two "
                "recombined light paths, coloured near zero path difference (white-light "
                "fringes) and washing to grey outward, shifting as a mirror scans. "
                "--frames scans the mirror.",
    renderer=_render,
)
