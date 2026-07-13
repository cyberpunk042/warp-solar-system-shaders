"""A prism — white light fanned into a spectrum.

Light slows in glass and bends at each surface (**Snell's law**), and because the
refractive index depends slightly on wavelength (**dispersion**) blue bends more than
red — so a triangular prism fans a white beam into the full **spectrum**. See
``docs/research/30-light-and-optics.md``. --frames drifts the beam.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import wavelength_rgb
from ..scene import Scene

_A = wp.constant(wp.vec2(0.0, 0.52))
_B = wp.constant(wp.vec2(-0.5, -0.4))
_C = wp.constant(wp.vec2(0.5, -0.4))
_EIN = wp.constant(wp.vec2(-0.22, 0.08))       # entry on the left face
_EOUT = wp.constant(wp.vec2(0.14, -0.16))      # exit on the right face


@wp.func
def _seg(p: wp.vec2, a: wp.vec2, b: wp.vec2) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.kernel
def prism_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                 width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.02, 0.02, 0.03)

    # incoming white beam + the refracted internal beam
    beam = wp.exp(-(_seg(p, wp.vec2(-1.6, 0.08), _EIN) / 0.012) ** 2.0)
    beam = beam + wp.exp(-(_seg(p, _EIN, _EOUT) / 0.012) ** 2.0)
    col = col + wp.vec3(1.0, 1.0, 1.0) * beam * 1.4

    # the dispersed output fan from the exit point
    d = p - _EOUT
    if d[0] > -0.02:
        ang = wp.atan2(d[1], d[0])
        ang0 = -0.42                                    # fan centre (down-right)
        hw = 0.30
        s = (ang - (ang0 - hw)) / (2.0 * hw)            # 0 (red, less bent) .. 1 (violet)
        if s > 0.0 and s < 1.0:
            nm = 645.0 - s * (645.0 - 400.0)
            band = wp.smoothstep(0.0, 0.06, s) * wp.smoothstep(1.0, 0.94, s)
            rad = wp.smoothstep(1.7, 0.05, wp.length(d))
            col = col + wavelength_rgb(nm) * band * rad * 1.5

    # glass prism: faint fill + bright edges
    dAB = _seg(p, _A, _B); dBC = _seg(p, _B, _C); dCA = _seg(p, _C, _A)
    edge = wp.min(dAB, wp.min(dBC, dCA))
    # inside test (barycentric sign)
    s1 = (_B[0] - _A[0]) * (p[1] - _A[1]) - (_B[1] - _A[1]) * (p[0] - _A[0])
    s2 = (_C[0] - _B[0]) * (p[1] - _B[1]) - (_C[1] - _B[1]) * (p[0] - _B[0])
    s3 = (_A[0] - _C[0]) * (p[1] - _C[1]) - (_A[1] - _C[1]) * (p[0] - _C[0])
    inside = (s1 < 0.0 and s2 < 0.0 and s3 < 0.0) or (s1 > 0.0 and s2 > 0.0 and s3 > 0.0)
    if inside:
        col = col + wp.vec3(0.10, 0.13, 0.18) * 0.5
    col = col + wp.vec3(0.6, 0.75, 0.95) * wp.exp(-(edge / 0.01) ** 2.0) * 0.7

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(prism_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="prism",
    description="A triangular glass prism fanning a white beam into the full spectrum — "
                "dispersion, because blue light bends more than red (Snell's law with a "
                "wavelength-dependent index). --frames drifts the beam.",
    renderer=_render,
)
