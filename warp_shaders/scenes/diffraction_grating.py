"""A diffraction grating — light split into spectral orders.

Thousands of fine parallel lines bend each wavelength to its own angle
(**d·sinθ = m·λ**): a white beam splits into a bright straight-through order (m=0)
and, on either side, first- and second-order **spectra** fanned out by wavelength —
the shimmer of a CD. See ``docs/research/30-light-and-optics.md``. --frames drifts
the beam.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import wavelength_rgb
from ..scene import Scene

_D = 2050.0        # grating pitch in "wavelength units" (sets the order spread)


@wp.kernel
def grating_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                   width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    col = wp.vec3(0.02, 0.02, 0.03)

    # incoming white beam to the grating at x=0
    if x < 0.0:
        col = col + wp.vec3(1.0, 1.0, 1.0) * wp.exp(-(y / 0.014) ** 2.0) \
            * wp.smoothstep(-1.6, -0.05, x) * 1.3

    # the grating itself: a thin striped bar
    if x > -0.05 and x < 0.0:
        stripe = 0.5 + 0.5 * wp.sin(y * 220.0)
        col = wp.vec3(0.35, 0.4, 0.5) * (0.4 + 0.6 * stripe)

    # diffracted orders on the far side
    if x > 0.0:
        r = wp.length(wp.vec2(x, y))
        sinf = y / r
        rad = wp.smoothstep(1.9, 0.05, r)
        # m = 0 straight-through (white)
        col = col + wp.vec3(1.0, 1.0, 1.0) * wp.exp(-(sinf / 0.02) ** 2.0) * rad * 1.2
        asf = wp.abs(sinf)
        # m = ±1
        lam1 = _D * asf
        if lam1 > 400.0 and lam1 < 690.0:
            band = wp.smoothstep(400.0, 430.0, lam1) * wp.smoothstep(690.0, 660.0, lam1)
            col = col + wavelength_rgb(lam1) * band * rad * 1.3
        # m = ±2 (fainter, wider)
        lam2 = _D * asf / 2.0
        if lam2 > 400.0 and lam2 < 690.0 and asf > 0.36:
            band = wp.smoothstep(400.0, 430.0, lam2) * wp.smoothstep(690.0, 660.0, lam2)
            col = col + wavelength_rgb(lam2) * band * rad * 0.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(grating_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="diffraction_grating",
    description="A diffraction grating splitting a white beam into spectral orders — a "
                "bright straight-through order (m=0) and first/second-order spectra fanned "
                "out by wavelength (d·sinθ = m·λ). --frames drifts the beam.",
    renderer=_render,
)
