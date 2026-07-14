"""warp_fold_chromo — the RTX 6000 Pro board folded into an X-shaped chromosome, in time.

The chromosome sibling of ``warp_fold_card``. The real `gpu_board` card is folded (its domain
accordion-folded and layer-stacked, exactly as for the cube) but the folded material is bounded
into the **X of a metaphase chromosome** — two crossing arms meeting at a centromere — instead of a
cube. So the whole graphics card condenses into a glowing X built from its own silicon and copper,
then unfolds back into the flat board. `time` drives compress → decompress.
"""

import math

import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import sd_box
from ..scene import Scene
from .gpu_board import board_map, board_shade
from .warp_fold_card import _warp

_MAXD = 40.0
_CYCLE = 10.0


@wp.func
def _orient(p: wp.vec3, time: float) -> wp.vec3:
    """Gentle rock (not a full spin) so the X keeps facing the camera as it forms."""
    a = 0.25 * wp.sin(time * 0.5)
    ca = wp.cos(a); sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _xshape(p: wp.vec3, fold: float) -> float:
    """The X of a chromosome: two crossing arms (±45° in the x-y plane), pinched at a centromere."""
    L = 10.0 * (1.0 - fold) + 1.75 * fold
    w = 10.0 * (1.0 - fold) + 0.42 * fold
    c = 0.7071
    s = 0.7071
    p1 = wp.vec3(c * p[0] + s * p[1], -s * p[0] + c * p[1], p[2])
    p2 = wp.vec3(c * p[0] - s * p[1], s * p[0] + c * p[1], p[2])
    # centromere constriction: arms narrow near the centre
    pinch = w * (0.55 + 0.45 * wp.min(wp.abs(p1[0]) + wp.abs(p2[0]), 1.0))
    a1 = sd_box(p1, wp.vec3(L, pinch, w))
    a2 = sd_box(p2, wp.vec3(L, pinch, w))
    return wp.min(a1, a2)


@wp.func
def _fmap(p: wp.vec3, time: float, fold: float) -> float:
    sp = _orient(p, time)
    board = board_map(_warp(sp, fold))
    return wp.max(board, _xshape(sp, fold))


@wp.func
def _fnormal(p: wp.vec3, time: float, fold: float) -> wp.vec3:
    e = 0.0012
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), time, fold) - _fmap(p - wp.vec3(e, 0.0, 0.0), time, fold)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), time, fold) - _fmap(p - wp.vec3(0.0, e, 0.0), time, fold)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), time, fold) - _fmap(p - wp.vec3(0.0, 0.0, e), time, fold)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, time: float, fold: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _fmap(p + n * hr, time, fold)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3, right: wp.vec3,
                   up: wp.vec3, width: int, height: int, time: float, tanfov: float, fold: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(260):
        p = eye + rd * t
        d = _fmap(p, time, fold)
        if d < 0.0007 * t + 0.0004:
            hit = 1
            break
        t += d * 0.8
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _fnormal(p, time, fold)
    ao = _fao(p, n, time, fold)
    q = _warp(_orient(p, time), fold)
    col = board_shade(q, n, rd, ao, time)
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    col = col + wp.vec3(0.3, 0.7, 1.0) * (seam * fold * 0.7)     # cool rim glow on the X
    img[i, j] = col


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _render(width, height, time, mouse, device):
    fold = _progress(time)
    az = 0.2 + float(mouse[0]) * 0.006
    el = 0.16 + 0.3 * (1.0 - fold)                  # start looking down at the flat board, rise to face the X
    dist = 8.6 * (1.0 - fold) + 6.2 * fold
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el) + 0.1,
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.05, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, float(time), tanfov, float(fold)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_fold_chromo",
    description="the real RTX 6000 Pro board folded into an X-shaped chromosome — the card's domain "
                "is folded and layer-stacked, then bounded into the two crossing arms of a metaphase "
                "chromosome built from its own silicon and copper, then unfolds flat again.",
    renderer=_render,
)
