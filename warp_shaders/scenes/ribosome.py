"""A ribosome — the protein factory reading the genetic code.

The ribosome reads messenger **RNA** three letters (a **codon**) at a time and links the
matching amino acids into a **protein** chain (translation). Two subunits clamp the
mRNA; **tRNAs** deliver amino acids; the growing protein threads out of the exit tunnel.
See ``docs/research/33-the-cell-up-close.md``. --frames translates the message.
"""

import math

import numpy as np

from ..engine import post
from ..fields.draw2d import draw_point
from ..scene import Scene

_CODON = [np.array(c, np.float32) for c in
          [(1.0, 0.4, 0.35), (0.4, 0.8, 0.5), (0.4, 0.6, 1.0), (1.0, 0.85, 0.4)]]


def _render(width, height, time, mouse, device):
    W, H = int(width), int(height)
    ax = W / H
    hdr = np.zeros((H, W, 3), np.float32)
    hdr[:, :] = np.array([0.03, 0.035, 0.05], np.float32)

    ys, xs = np.mgrid[0:H, 0:W]
    wx = (xs / W * 2.0 - 1.0) * ax
    wy = 1.0 - ys / H * 2.0
    rng = np.random.default_rng(5)
    speck = rng.random((H, W)).astype(np.float32)

    def blob(cx, cy, rx, ry, color):
        e = ((wx - cx) / rx) ** 2 + ((wy - cy) / ry) ** 2
        m = e < 1.0
        shade = (0.55 + 0.5 * np.sqrt(np.clip(1.0 - e, 0, 1)))[m]
        tex = 0.7 + 0.5 * speck[m]
        hdr[m] = np.asarray(color, np.float32)[None, :] * (shade * tex)[:, None]

    # large (top) + small (bottom) subunits clamping the mRNA groove
    blob(0.0, 0.28, 0.9, 0.62, (0.52, 0.46, 0.4))
    blob(0.05, -0.34, 0.72, 0.42, (0.62, 0.52, 0.44))

    # mRNA threading through the interface (codon-coloured beads, scrolling)
    scroll = (time * 0.18) % 0.14
    for k in range(-14, 15):
        mx = k * 0.14 + scroll
        if abs(mx) > 1.5 * ax:
            continue
        my = -0.04 + 0.015 * math.sin(mx * 5.0)
        draw_point(hdr, (mx, my), _CODON[(k) % 4] * 1.2, max(2.5, W * 0.008), ax)

    # tRNA at the A-site, delivering an amino acid
    draw_point(hdr, (0.16, -0.02), (1.0, 0.9, 0.5), max(3.0, W * 0.012), ax)
    draw_point(hdr, (0.22, 0.12), (0.9, 0.5, 0.9), max(3.0, W * 0.013), ax)

    # the growing protein chain threading out of the exit tunnel (rainbow N->C)
    nprot = min(5 + int(time * 1.6), 24)
    ex = np.array([-0.15, 0.82])
    for m in range(nprot):
        t = m / 23.0
        p = ex + np.array([-1.15 * t, 0.16 * math.sin(t * 6.0) - 0.02 * t])
        hue = m / max(nprot - 1, 1)
        c = np.array([0.5 + 0.5 * math.cos(hue * 6.28),
                      0.5 + 0.5 * math.cos(hue * 6.28 + 2.1),
                      0.5 + 0.5 * math.cos(hue * 6.28 + 4.2)], np.float32)
        draw_point(hdr, tuple(p), c * 1.1, max(2.5, W * 0.009), ax)

    r = max(2, int(min(W, H) * 0.006))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="ribosome",
    description="A ribosome translating — two subunits clamping a codon-coloured mRNA "
                "strand, a tRNA delivering an amino acid at the A-site, and the growing "
                "protein chain threading out of the exit tunnel. --frames translates the "
                "message.",
    renderer=_render,
)
