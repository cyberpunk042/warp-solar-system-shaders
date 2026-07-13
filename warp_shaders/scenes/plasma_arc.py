"""A plasma arc — the fourth state of matter, glowing.

Ionise a gas — tear the electrons from the nuclei — and it becomes a **plasma**: a
glowing, conductive soup. A Tesla-coil arc leaps between electrodes as a branching
fractal filament, the air itself turned to plasma along the discharge path. See
``docs/research/31-states-of-matter.md``. --frames re-strikes the arc.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


def _bolt(rng, a, b, disp, levels):
    pts = [np.array(a, np.float32), np.array(b, np.float32)]
    for _ in range(levels):
        out = []
        for i in range(len(pts) - 1):
            p0, p1 = pts[i], pts[i + 1]
            mid = 0.5 * (p0 + p1)
            d = p1 - p0
            perp = np.array([-d[1], d[0]], np.float32)
            perp /= (np.linalg.norm(perp) + 1e-6)
            mid = mid + perp * (rng.random() - 0.5) * disp
            out.append(p0); out.append(mid)
        out.append(pts[-1])
        pts = out
        disp *= 0.52
    return pts


def _segments(frame):
    rng = np.random.default_rng(frame % 100000)
    pts = _bolt(rng, (0.0, 0.92), (0.0, -0.92), 0.55, 6)
    segs = []
    for i in range(len(pts) - 1):
        segs.append((pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]))
    # a few branches forking off the main channel
    for _ in range(4):
        k = rng.integers(len(pts) // 4, 3 * len(pts) // 4)
        start = pts[int(k)]
        end = start + np.array([(rng.random() - 0.5) * 1.3, -rng.random() * 0.7], np.float32)
        bp = _bolt(rng, tuple(start), tuple(end), 0.28, 4)
        for i in range(len(bp) - 1):
            segs.append((bp[i][0], bp[i][1], bp[i + 1][0], bp[i + 1][1]))
    return np.array(segs, np.float32)


@wp.func
def _segd(p: wp.vec2, a: wp.vec2, b: wp.vec2) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.kernel
def arc_kernel(img: wp.array2d(dtype=wp.vec3), segs: wp.array(dtype=wp.vec4),
               nseg: int, aspect: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.015, 0.015, 0.03)

    dmin = float(1e9)
    for s in range(nseg):
        seg = segs[s]
        d = _segd(p, wp.vec2(seg[0], seg[1]), wp.vec2(seg[2], seg[3]))
        dmin = wp.min(dmin, d)
    glow = wp.exp(-(dmin / 0.05) * (dmin / 0.05))
    core = wp.exp(-(dmin / 0.008) * (dmin / 0.008))
    col = col + wp.vec3(0.4, 0.6, 1.0) * glow * 1.1 + wp.vec3(0.9, 0.95, 1.0) * core * 1.6

    # electrodes (top + bottom) with a hot terminal glow
    et = wp.length(p - wp.vec2(0.0, 0.95))
    eb = wp.length(p - wp.vec2(0.0, -0.95))
    col = col + wp.vec3(0.7, 0.8, 1.0) * (wp.exp(-(et / 0.08) ** 2.0) + wp.exp(-(eb / 0.08) ** 2.0)) * 1.2

    img[i, j] = col


def _render(width, height, time, mouse, device):
    frame = int(time * 11.0)
    segs_np = _segments(frame)
    segs = wp.array(segs_np, dtype=wp.vec4, device=device)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(arc_kernel, dim=(height, width),
              inputs=[img, segs, int(segs_np.shape[0]), float(width / height),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.6, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="plasma_arc",
    description="A plasma arc — a Tesla-coil discharge leaping between electrodes as a "
                "branching fractal filament, the air ionised into glowing plasma (the "
                "fourth state of matter) along the channel. --frames re-strikes the arc.",
    renderer=_render,
)
