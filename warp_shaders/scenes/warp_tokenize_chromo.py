"""warp_tokenize_chromo — C3: tokens weave into a tight mesh chromosome, in its storage cube.

Operator spec (verbatim, 2026-07-16): *"a really tightly bounded mesh of those tokens like becoming a mini
chromosome so much its compressed together, like weaving a token after forming DNA like strain that keep
and keep interlocking to form a unite shape / wall / exterior. The scan can be different near transparent.
the goal is to have a token of color per elements similar to the merge / second compression. and I want a
real chromosome at the end... and little tensor shape like two arms and two legs that could move in a cube
with super density. it can be so compressed that it has a representation of a larger polygone / cube or
rectangle around itself in the compression storage."*

C3 as a full reversible process on the real `gpu_board`:

  1. **The card**, then a **near-transparent scan** sweeps it — every element takes **one colour = its
     `warp_compress` token** (the same tokens as the merge, C1).
  2. **Weave** — the card's material **coils and weaves**, layer interlocking on layer, tightening into a
     **real metaphase chromosome** (the two-arms/two-legs X). It is not the raw board any more: it is a
     **dense woven mesh of coloured tokens** — each cell of the mesh coloured by the card element it came
     from (mapped straight through the coil), packed so tight it forms one solid shape / wall / exterior.
  3. **The storage cube** — a wire **bounding cube** draws itself around the chromosome: the compact
     polygon that holds the super-dense token-mesh for decompression.
  4. **Reverse** — it all runs backwards, the chromosome unwinding to the flat card. `time` runs the whole
     cycle, then loops.

Codec behind it: `warp_compress.tokenchromo` (lossless, verified).
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import op_smooth_union, sd_capsule, sd_sphere, sd_round_box
from ..scene import Scene
from .gpu_board import board_map
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card, _BB

_MAXD = 40.0
_CYCLE = 15.0
_BLOCK = 5
_MESH = 26.0                     # token-mesh cell frequency (how fine the woven cells are)


def _build():
    """the card's per-block token volume (index grid) — one token/colour per element, for the mesh."""
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)          # index: (nbx, nby, nbz) token per block
    return np.ascontiguousarray(index.astype(np.int32))


_INDEX = _build()
_NBX, _NBY, _NBZ = _INDEX.shape


# ------------------------------------------------------------------ real-material coil (from fold_chromo)
@wp.func
def _tri(a: float, f: float) -> float:
    a = wp.abs(a)
    a = a - 2.0 * f * wp.floor(a / (2.0 * f))
    if a > f:
        a = 2.0 * f - a
    return a


@wp.func
def _fill(p: wp.vec3, fold: float) -> wp.vec3:
    ang = fold * 1.5708
    ca = wp.cos(ang); sa = wp.sin(ang)
    q = wp.vec3(p[0], ca * p[1] - sa * p[2], sa * p[1] + ca * p[2])
    per = 6.0 * (1.0 - fold) + 1.15 * fold
    qx = _tri(q[0] + 3.7 * fold, per) - 0.5 * per * fold
    qz = _tri(q[2] + 1.5 * fold, per) - 0.5 * per * fold
    lh = 100.0 * (1.0 - fold) + 0.5 * fold
    qy = q[1] - lh * wp.round(q[1] / lh)
    return wp.vec3(qx, qy, qz)


@wp.func
def _xshape(p: wp.vec3, fold: float) -> float:
    L = 1.55; r = 0.5; a0 = 0.26; dx = 0.60; dy = 0.80
    ur = sd_capsule(p, wp.vec3(a0 * dx, a0 * dy, 0.0), wp.vec3(L * dx, L * dy, 0.0), r)
    ul = sd_capsule(p, wp.vec3(-a0 * dx, a0 * dy, 0.0), wp.vec3(-L * dx, L * dy, 0.0), r)
    lr = sd_capsule(p, wp.vec3(a0 * dx, -a0 * dy, 0.0), wp.vec3(L * dx, -L * dy, 0.0), r)
    ll = sd_capsule(p, wp.vec3(-a0 * dx, -a0 * dy, 0.0), wp.vec3(-L * dx, -L * dy, 0.0), r)
    x = op_smooth_union(ur, ul, 0.28)
    x = op_smooth_union(x, lr, 0.28)
    x = op_smooth_union(x, ll, 0.28)
    x = op_smooth_union(x, sd_sphere(p, 0.34), 0.22)
    slab = sd_round_box(p, wp.vec3(3.7, 0.32, 1.5), 0.1)
    e = fold * fold * (3.0 - 2.0 * fold)
    return slab * (1.0 - e) + x * e


@wp.func
def _cfmap(p: wp.vec3, fold: float) -> float:
    return wp.max(board_map(_fill(p, fold)), _xshape(p, fold))


@wp.func
def _boxframe(p: wp.vec3, b: wp.vec3, e: float) -> float:
    q = wp.vec3(wp.abs(p[0]) - b[0], wp.abs(p[1]) - b[1], wp.abs(p[2]) - b[2])
    qx = wp.abs(q[0] + e) - e
    qy = wp.abs(q[1] + e) - e
    qz = wp.abs(q[2] + e) - e
    d1 = wp.length(wp.vec3(wp.max(qx, 0.0), wp.max(qy, 0.0), wp.max(q[2], 0.0))) + wp.min(wp.max(qx, wp.max(qy, q[2])), 0.0)
    d2 = wp.length(wp.vec3(wp.max(qx, 0.0), wp.max(q[1], 0.0), wp.max(qz, 0.0))) + wp.min(wp.max(qx, wp.max(q[1], qz)), 0.0)
    d3 = wp.length(wp.vec3(wp.max(q[0], 0.0), wp.max(qy, 0.0), wp.max(qz, 0.0))) + wp.min(wp.max(q[0], wp.max(qy, qz)), 0.0)
    return wp.min(wp.min(d1, d2), d3)


_BOXB = wp.constant(wp.vec3(1.55, 2.0, 0.85))    # bounding-cube half-extents around the chromosome


@wp.func
def _map(p: wp.vec3, fold: float, cagew: float) -> float:
    d = _cfmap(p, fold)
    if cagew > 0.01:
        d = wp.min(d, _boxframe(p, _BOXB, 0.02))
    return d


@wp.func
def _normal(p: wp.vec3, fold: float, cagew: float) -> wp.vec3:
    e = 0.0013
    dx = _map(p + wp.vec3(e, 0.0, 0.0), fold, cagew) - _map(p - wp.vec3(e, 0.0, 0.0), fold, cagew)
    dy = _map(p + wp.vec3(0.0, e, 0.0), fold, cagew) - _map(p - wp.vec3(0.0, e, 0.0), fold, cagew)
    dz = _map(p + wp.vec3(0.0, 0.0, e), fold, cagew) - _map(p - wp.vec3(0.0, 0.0, e), fold, cagew)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, fold: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _cfmap(p + n * hr, fold)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.func
def _hue(h: float) -> wp.vec3:
    r = wp.clamp(wp.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = wp.clamp(2.0 - wp.abs(h * 6.0 - 2.0), 0.0, 1.0)
    bb = wp.clamp(2.0 - wp.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return wp.vec3(r, g, bb)


@wp.func
def _tokcolor(tid: int) -> wp.vec3:
    if tid < 0:
        return wp.vec3(0.10, 0.13, 0.12)
    h = (float(tid) * 0.61803) % 1.0
    return _hue(h)


@wp.func
def _tok_at(q: wp.vec3, index: wp.array3d(dtype=wp.int32), nbx: int, nby: int, nbz: int) -> int:
    fx = (q[0] - (-3.7)) / 7.4
    fy = (q[1] - (-0.14)) / 0.44
    fz = (q[2] - (-1.5)) / 3.0
    bi = int(wp.clamp(fx * float(nbx), 0.0, float(nbx - 1)))
    by = int(wp.clamp(fy * float(nby), 0.0, float(nby - 1)))
    bk = int(wp.clamp(fz * float(nbz), 0.0, float(nbz - 1)))
    return index[bi, by, bk]


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), index: wp.array3d(dtype=wp.int32),
                   nbx: int, nby: int, nbz: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, fold: float, cagew: float, scanx: float, tokamt: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(240):
        p = eye + rd * t
        d = _map(p, fold, cagew)
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
    # the wire storage-cube?
    if cagew > 0.01 and _boxframe(p, _BOXB, 0.02) < _cfmap(p, fold) + 0.0005:
        img[i, j] = wp.vec3(0.35, 0.85, 1.0) * (0.8 * cagew)
        return

    n = _normal(p, fold, cagew)
    ao = _ao(p, n, fold)
    q = _fill(p, fold)                                       # board-local coord this material came from
    tid = _tok_at(q, index, nbx, nby, nbz)
    tc = _tokcolor(tid)

    # woven token-mesh: colour per element (token) + tight interlocking cell seams in board space
    mx = wp.abs(wp.sin(q[0] * _MESH))
    my = wp.abs(wp.sin((q[1] * 3.0) * _MESH))
    mz = wp.abs(wp.sin(q[2] * _MESH))
    weave = wp.min(wp.min(mx, my), mz)
    cell = wp.clamp(weave * 6.0, 0.30, 1.0)                  # dark seams -> woven cells
    lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.5))), 0.2, 1.0)
    base = tc * ((0.45 + 0.6 * lit) * ao * cell)
    # near-transparent scan sweeping down the card early on
    band = wp.abs(q[0] - scanx)
    scan = float(0.0)
    if band < 0.25:
        scan = (1.0 - band / 0.25) * (1.0 - tokamt)
    col = base * tokamt + wp.vec3(0.10, 0.14, 0.13) * (1.0 - tokamt)   # fade in the token colour as it weaves
    col = col + wp.vec3(0.4, 0.9, 1.0) * (scan * 0.5)
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    col = col + tc * (seam * fold * 0.4)                     # rim as it tightens into the X
    img[i, j] = col


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _state(time):
    """(fold, cage_w, scan_x, tok_amt) — card → scan → weave into token-mesh X + cube → reverse."""
    u = (float(time) % _CYCLE) / _CYCLE
    if u < 0.12:                                    # the flat card (real board colours, no tokens yet)
        return 0.0, 0.0, -3.7, 0.0
    if u < 0.30:                                    # near-transparent scan: tokens colour in per element
        f = (u - 0.12) / 0.18
        return 0.0, 0.0, -3.7 + 7.4 * _smooth(f), _smooth(f)
    if u < 0.60:                                    # weave: the card coils/tightens into the chromosome X
        f = (u - 0.30) / 0.30
        return _smooth(f), 0.0, 4.0, 1.0
    if u < 0.72:                                    # the storage cube draws itself around the dense mesh
        f = (u - 0.60) / 0.12
        return 1.0, _smooth(f), 4.0, 1.0
    if u < 0.88:                                    # hold the compressed chromosome in its cube
        return 1.0, 1.0, 4.0, 1.0
    # reverse: cube fades, chromosome unwinds to the flat card
    f = (u - 0.88) / 0.12
    return 1.0 - _smooth(f), 1.0 - _smooth(f), 4.0, 1.0 - 0.5 * _smooth(f)


def _render(width, height, time, mouse, device):
    fold, cagew, scanx, tokamt = _state(time)
    index = wp.array3d(_INDEX, dtype=wp.int32, device=device)

    az = 0.30 + 0.5 * fold + 0.08 * math.sin(time * 0.2) + float(mouse[0]) * 0.006
    el = 0.34 * (1.0 - fold) + 0.14 * fold
    dist = 9.4 * (1.0 - fold) + 6.6 * fold
    tgt = wp.vec3(0.0, 0.15 + 0.15 * fold, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, index, _NBX, _NBY, _NBZ, eye, fwd, right, up, width, height,
                      float(time), tanfov, float(fold), float(cagew), float(scanx), float(tokamt)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.15, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3 as a full reversible process: a near-transparent scan gives every element one "
                "warp_compress token colour, the card weaves and coils into a real metaphase chromosome "
                "(the two-arm/two-leg X) built as a tight mesh of coloured tokens, and a wire storage-cube "
                "draws itself around the super-dense mesh — then it unwinds back to the flat card.",
    renderer=_render,
)
