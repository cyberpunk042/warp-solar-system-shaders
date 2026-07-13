"""The periodic table — all the elements, arranged and block-coloured.

Every element in the canonical 18-column layout (with the f-block lanthanides &
actinides below), each a glowing tile coloured by block: **s** (warm red), **p**
(teal), **d** (gold), **f** (magenta), noble gases (violet), hydrogen (green).
The shape of the table is instantly recognisable. See
``docs/research/22-chemistry-and-molecules.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene

# block ids → colours
_COLS = [
    (0.92, 0.42, 0.36),   # 0 s-block
    (0.35, 0.75, 0.88),   # 1 p-block
    (0.96, 0.76, 0.3),    # 2 d-block
    (0.85, 0.42, 0.86),   # 3 f-block
    (0.6, 0.46, 0.96),    # 4 noble gas
    (0.5, 0.9, 0.55),     # 5 hydrogen
]


def _grid():
    g = -np.ones((9, 18), np.int32)
    g[0, 0] = 5; g[0, 17] = 4                          # H, He
    for r in (1, 2):                                   # periods 2-3
        g[r, 0] = 0; g[r, 1] = 0
        for c in range(12, 18):
            g[r, c] = 1
        g[r, 17] = 4
    for r in (3, 4):                                   # periods 4-5
        g[r, 0] = 0; g[r, 1] = 0
        for c in range(2, 12):
            g[r, c] = 2
        for c in range(12, 18):
            g[r, c] = 1
        g[r, 17] = 4
    for r in (5, 6):                                   # periods 6-7
        g[r, 0] = 0; g[r, 1] = 0
        g[r, 2] = 3                                    # La / Ac (f-block marker)
        for c in range(3, 12):
            g[r, c] = 2
        for c in range(12, 18):
            g[r, c] = 1
        g[r, 17] = 4
    for c in range(2, 17):                             # f-block rows
        g[7, c] = 3; g[8, c] = 3
    return g


@wp.kernel
def table_kernel(img: wp.array2d(dtype=wp.vec3), grid: wp.array2d(dtype=wp.int32),
                 cols: wp.array(dtype=wp.vec3), ncol: int, nrow: int,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    x = (float(j) + 0.5) / float(width)                # 0..1
    y = (float(i) + 0.5) / float(height)
    # map into grid space with a margin; rows 0..8 top→bottom, small gap before f-block
    gx = x * float(ncol) / 0.94 - 0.3
    gy = y * (float(nrow) + 0.6) / 0.94 - 0.2
    if gy > 7.0:
        gy = gy - 0.6                                  # visual gap above the f-block
    col = wp.vec3(0.02, 0.025, 0.045)
    cc = int(wp.floor(gx))
    rr = int(wp.floor(gy))
    if cc >= 0 and cc < ncol and rr >= 0 and rr < nrow:
        b = grid[rr, cc]
        if b >= 0:
            fx = gx - wp.floor(gx)
            fy = gy - wp.floor(gy)
            # rounded tile with a soft bevel
            dx = wp.abs(fx - 0.5)
            dy = wp.abs(fy - 0.5)
            m = wp.max(dx, dy)
            if m < 0.42:
                base = cols[b]
                bevel = 1.0 - 1.1 * wp.max(dx, dy)
                glow = 0.75 + 0.25 * wp.sin(time * 1.5 + float(cc) + float(rr))
                edge = wp.smoothstep(0.42, 0.34, m)
                col = base * (bevel * glow) * edge + base * 0.15
    img[i, j] = col


def _render(width, height, time, mouse, device):
    g = _grid()
    nrow, ncol = g.shape
    cols = np.array(_COLS, np.float32)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(table_kernel, dim=(height, width),
              inputs=[img, wp.array(g, dtype=wp.int32, device=device),
                      wp.array(cols, dtype=wp.vec3, device=device), int(ncol),
                      int(nrow), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=0.9, strength=0.3, radius=3, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.08)


SCENE = Scene(
    name="periodic_table",
    description="The periodic table — every element in the 18-column layout "
                "(f-block below), tiles coloured by block: s (red), p (teal), "
                "d (gold), f (magenta), noble (violet), H (green).",
    renderer=_render,
)
