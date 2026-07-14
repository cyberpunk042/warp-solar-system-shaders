"""warp_fold_chromo — the RTX 6000 Pro board folded into a metaphase chromosome (the X), in time.

The chromosome sibling of ``warp_fold_card``. The real `gpu_board` card lifts off the bench and
**wraps into the X of a metaphase chromosome** — two sister chromatids, four plump rounded arms
joined at a pinched **centromere**, exactly the classic textbook shape. The arms are fat rounded
capsules blended at a narrow waist with a centromere bead; the card's own board material (green
solder mask, gold routing, GDDR7) fills them, folded and packed. `time` drives the wrap forward
(compress) and back (decompress) — the flat board and the chromosome are the two ends of one fold.
"""

import math

import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import op_smooth_union, sd_capsule, sd_round_box, sd_sphere
from ..scene import Scene
from .gpu_board import board_map, board_shade

_MAXD = 40.0
_CYCLE = 10.0


@wp.func
def _tri(a: float, f: float) -> float:
    """Accordion fold: reflect any coordinate into ``[0, f]`` at each ``f`` so the long board tiles."""
    a = wp.abs(a)
    a = a - 2.0 * f * wp.floor(a / (2.0 * f))
    if a > f:
        a = 2.0 * f - a
    return a


@wp.func
def _fill(p: wp.vec3, fold: float) -> wp.vec3:
    """Map a point in the chromosome frame back to board-local coords: stand the flat board up into
    the X's plane (rotate about x by fold*90 deg) and accordion-fold so the card packs the arms."""
    ang = fold * 1.5708
    ca = wp.cos(ang)
    sa = wp.sin(ang)
    q = wp.vec3(p[0], ca * p[1] - sa * p[2], sa * p[1] + ca * p[2])   # Rx: flat (x,z) -> standing (x,y)
    per = 6.0 * (1.0 - fold) + 1.15 * fold
    qx = _tri(q[0] + 3.7 * fold, per) - 0.5 * per * fold
    qz = _tri(q[2] + 1.5 * fold, per) - 0.5 * per * fold
    lh = 100.0 * (1.0 - fold) + 0.5 * fold
    qy = q[1] - lh * wp.round(q[1] / lh)
    return wp.vec3(qx, qy, qz)


@wp.func
def _xshape(p: wp.vec3, fold: float) -> float:
    """The metaphase chromosome: four fat rounded arms (capsules) blended into a pinched centromere,
    plus a centromere bead — the classic X. At fold=0 it opens out to a flat slab over the board."""
    L = 1.55                                   # arm length
    r = 0.5                                    # arm fatness (round caps -> plump lobes)
    a0 = 0.26                                  # gap left near the centre -> the waist
    dx = 0.60
    dy = 0.80                                  # arm direction (a touch taller than wide)
    ur = sd_capsule(p, wp.vec3(a0 * dx, a0 * dy, 0.0), wp.vec3(L * dx, L * dy, 0.0), r)
    ul = sd_capsule(p, wp.vec3(-a0 * dx, a0 * dy, 0.0), wp.vec3(-L * dx, L * dy, 0.0), r)
    lr = sd_capsule(p, wp.vec3(a0 * dx, -a0 * dy, 0.0), wp.vec3(L * dx, -L * dy, 0.0), r)
    ll = sd_capsule(p, wp.vec3(-a0 * dx, -a0 * dy, 0.0), wp.vec3(-L * dx, -L * dy, 0.0), r)
    x = op_smooth_union(ur, ul, 0.28)
    x = op_smooth_union(x, lr, 0.28)
    x = op_smooth_union(x, ll, 0.28)
    x = op_smooth_union(x, sd_sphere(p, 0.34), 0.22)          # centromere bead
    slab = sd_round_box(p, wp.vec3(3.7, 0.32, 1.5), 0.1)      # the un-wrapped flat card
    e = fold * fold * (3.0 - 2.0 * fold)
    return slab * (1.0 - e) + x * e


@wp.func
def _fmap(p: wp.vec3, fold: float) -> float:
    return wp.max(board_map(_fill(p, fold)), _xshape(p, fold))


@wp.func
def _fnormal(p: wp.vec3, fold: float) -> wp.vec3:
    e = 0.0013
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), fold) - _fmap(p - wp.vec3(e, 0.0, 0.0), fold)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), fold) - _fmap(p - wp.vec3(0.0, e, 0.0), fold)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), fold) - _fmap(p - wp.vec3(0.0, 0.0, e), fold)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, fold: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _fmap(p + n * hr, fold)
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
    for _ in range(300):
        p = eye + rd * t
        d = _fmap(p, fold)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.7
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _fnormal(p, fold)
    ao = _fao(p, n, fold)
    q = _fill(p, fold)
    col = board_shade(q, n, rd, ao, time)
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    col = col + wp.vec3(0.3, 0.7, 1.0) * (seam * fold * 0.6)     # cool rim glow on the X
    img[i, j] = col


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _render(width, height, time, mouse, device):
    fold = _progress(time)
    az = 0.15 + float(mouse[0]) * 0.006
    el = 0.14 + 0.42 * (1.0 - fold)                 # look down at the flat board, rise to face the X
    dist = 9.0 * (1.0 - fold) + 5.4 * fold
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
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
    description="the real RTX 6000 Pro board (gpu_board) wrapped into a metaphase chromosome — four "
                "plump rounded arms joined at a pinched centromere, the classic X, filled with the "
                "card's own folded board material, then unwrapping back to the flat board.",
    renderer=_render,
)
