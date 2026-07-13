"""A mitochondrion — the powerhouse of the cell.

The mitochondrion burns food with oxygen to make **ATP**, the cell's energy currency.
A smooth **outer membrane** wraps a deeply folded **inner membrane** — the **cristae** —
whose huge surface area carries the electron-transport chain. Here a cross-section: the
double membrane, the folded cristae, and ATP glowing in the matrix. See
``docs/research/33-the-cell-up-close.md``. --frames pulses the energy.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.kernel
def mito_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, time: float,
                width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0

    a = 1.35
    b = 0.72
    # rough signed distance to the bean-shaped outer membrane
    rr = wp.length(wp.vec2(x / a, y / b))
    d = (rr - 1.0) * b
    col = wp.vec3(0.02, 0.03, 0.05)                    # cytoplasm outside

    if d < 0.0:
        # matrix (interior), warm granular
        gran = fbm3(wp.vec3(x * 6.0, y * 6.0, 1.0), 3)
        col = wp.vec3(0.5, 0.32, 0.3) * (0.6 + 0.5 * gran)
        # cristae: folded inner-membrane sheets
        phase = y * 15.0 + 2.2 * wp.sin(x * 4.5 + time * 0.5)
        fold = wp.abs(wp.sin(phase))
        crista = wp.smoothstep(0.82, 0.98, fold) * wp.smoothstep(-0.02, -0.14, d)
        col = col + wp.vec3(0.7, 0.45, 0.65) * crista * 0.8
        # ATP energy glow travelling along the cristae (round granules)
        cxx = wp.floor(x * 12.0)
        cyy = wp.floor(y * 12.0 + time * 2.0)
        if hash21(wp.vec2(cxx, cyy)) > 0.93:
            fxx = x * 12.0 - cxx - 0.5
            fyy = y * 12.0 - (wp.floor(y * 12.0)) - 0.5
            dot = wp.exp(-(fxx * fxx + fyy * fyy) * 10.0)
            col = col + wp.vec3(1.0, 0.85, 0.4) * dot * crista * 2.0

    # the double membrane (two close lines)
    col = col + wp.vec3(0.85, 0.55, 0.4) * wp.exp(-((d + 0.02) / 0.016) ** 2.0) * 0.9
    col = col + wp.vec3(0.9, 0.65, 0.5) * wp.exp(-((d - 0.05) / 0.016) ** 2.0) * 0.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(mito_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.35, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="mitochondrion",
    description="A mitochondrion cross-section — the smooth outer membrane, the deeply "
                "folded inner-membrane cristae carrying the electron-transport chain, and "
                "ATP energy glowing in the matrix. The powerhouse of the cell. "
                "--frames pulses the energy.",
    renderer=_render,
)
