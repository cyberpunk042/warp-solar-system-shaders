"""warp_tokenize_chromo — C3: colour-tokens thread a DNA strand that condenses into the X chromosome.

Operator spec (verbatim, 2026-07-16): *"the scan can be different near transparent, the goal is to have a
token of color per elements similar to the merge / second compression ... weaving a token after forming DNA
like strain that keep and keep interlocking to form a unite shape / wall / exterior ... a really tightly
bounded mesh of those tokens like becoming a mini chromosome so much its compressed together ... I want a
real chromosome at the end ... little tensor shape like two arms and two legs that could move in a cube with
super density ... a representation of a larger polygone / cube or rectangle around itself in the compression
storage."*

C3 as the real biology of packing information — DNA → condensed chromosome — run on the `gpu_board`:

  1. **The card**, then a **near-transparent scan** sweeps it: every element takes **one colour = its
     `warp_compress` token** (the same tokens as the merge, C1) — a token of colour per element.
  2. **DNA strand** — the tokens thread onto a **double helix** laid out along the card (two strands
     winding around each other, coloured by the element each stretch came from).
  3. **Condense** — the strand **keeps coiling and interlocking**, winding tighter and tighter (more turns,
     fatter fibre) as it folds from the extended horizontal DNA into the compact **metaphase chromosome**:
     the real **X** — two arms, two legs — joined at the centromere. Super dense; a woven token mesh.
  4. **The storage cube** — a wire **bounding cube** draws itself around the chromosome: the compact
     polygon that holds the super-dense token-strand for decompression.
  5. **Reverse** — it all runs backwards, the chromosome unwinding to the flat card. `time` runs the whole
     cycle, then loops.

Codec behind it: `warp_compress.tokenchromo` (lossless, verified).
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import op_smooth_union, sd_sphere
from ..scene import Scene
from .gpu_board import board_map
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card

_MAXD = 40.0
_CYCLE = 15.0
_BLOCK = 5

# --- the X-chromosome arm endpoints (fold=1, condensed) and their flat DNA layout (fold=0) -----------
# Each of the four arms is a double-helix segment. At fold=1 they splay into the X (two arms up, two down),
# joined at the centromere. At fold=0 they line up head-to-tail into ONE extended strand across the card.
_XB_UL = wp.constant(wp.vec3(-0.16, 0.22, 0.0)); _XT_UL = wp.constant(wp.vec3(-0.98, 1.34, 0.0))
_XB_LL = wp.constant(wp.vec3(-0.16, -0.22, 0.0)); _XT_LL = wp.constant(wp.vec3(-0.98, -1.34, 0.0))
_XB_LR = wp.constant(wp.vec3(0.16, -0.22, 0.0)); _XT_LR = wp.constant(wp.vec3(0.98, -1.34, 0.0))
_XB_UR = wp.constant(wp.vec3(0.16, 0.22, 0.0)); _XT_UR = wp.constant(wp.vec3(0.98, 1.34, 0.0))
# flat DNA: four card-length quarters, left -> right, so tokens span the whole board
_FB_UL = wp.constant(wp.vec3(-3.7, 0.0, 0.0)); _FT_UL = wp.constant(wp.vec3(-1.85, 0.0, 0.0))
_FB_LL = wp.constant(wp.vec3(-1.85, 0.0, 0.0)); _FT_LL = wp.constant(wp.vec3(0.0, 0.0, 0.0))
_FB_LR = wp.constant(wp.vec3(0.0, 0.0, 0.0)); _FT_LR = wp.constant(wp.vec3(1.85, 0.0, 0.0))
_FB_UR = wp.constant(wp.vec3(1.85, 0.0, 0.0)); _FT_UR = wp.constant(wp.vec3(3.7, 0.0, 0.0))
_BOXB = wp.constant(wp.vec3(1.45, 1.75, 0.75))    # bounding-cube half-extents around the chromosome


def _build():
    """the card's per-block token volume (index grid) — one token/colour per element."""
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)          # index: (nbx, nby, nbz) token per block
    return np.ascontiguousarray(index.astype(np.int32))


_INDEX = _build()
_NBX, _NBY, _NBZ = _INDEX.shape


@wp.func
def _perp(a: wp.vec3) -> wp.vec3:
    ref = wp.vec3(0.0, 0.0, 1.0)
    if wp.abs(a[2]) > 0.9:
        ref = wp.vec3(1.0, 0.0, 0.0)
    return wp.normalize(wp.cross(a, ref))


@wp.func
def _arm(p: wp.vec3, fb: wp.vec3, ft: wp.vec3, xb: wp.vec3, xt: wp.vec3,
         e: float, rr: float, tr: float, om: float) -> wp.vec2:
    """double-helix arm. returns (distance, source-card-x). fb/ft = flat DNA layout (token source);
    xb/xt = condensed X-arm endpoints. `e` morphs geometry flat->X; token always from the flat source."""
    base = fb * (1.0 - e) + xb * e
    tip = ft * (1.0 - e) + xt * e
    ax = tip - base
    L = wp.length(ax)
    axn = ax / wp.max(L, 0.0001)
    w = p - base
    sa = wp.dot(w, axn)
    sc = wp.clamp(sa, 0.0, L)
    perp = w - axn * sc
    uu = _perp(axn)
    vv = wp.cross(axn, uu)
    pu = wp.dot(perp, uu)
    pv = wp.dot(perp, vv)
    rad = wp.sqrt(pu * pu + pv * pv)
    ang = wp.atan2(pv, pu)
    tw = om * sc
    d0 = ang - tw
    d0 = d0 - 6.2831853 * wp.round(d0 / 6.2831853)
    d1 = ang - tw - 3.14159265
    d1 = d1 - 6.2831853 * wp.round(d1 / 6.2831853)
    hr0 = wp.sqrt((rad - rr) * (rad - rr) + (rr * d0) * (rr * d0))
    hr1 = wp.sqrt((rad - rr) * (rad - rr) + (rr * d1) * (rr * d1))
    axo = sa - sc
    e0 = wp.sqrt(hr0 * hr0 + axo * axo) - tr
    e1 = wp.sqrt(hr1 * hr1 + axo * axo) - tr
    dstr = wp.min(e0, e1)
    cardx = fb[0] + (ft[0] - fb[0]) * (sc / wp.max(L, 0.0001))
    return wp.vec2(dstr, cardx)


@wp.func
def _chromo(p: wp.vec3, e: float) -> float:
    """the whole double-helix chromosome at morph `e`: 4 woven arms + centromere."""
    om = 5.0 + 7.0 * e                              # winds tighter (more turns) as it condenses
    rr = 0.20 - 0.05 * e                            # helix radius shrinks
    tr = 0.070 + 0.055 * e                          # fibre fattens -> super-dense mesh
    d = _arm(p, _FB_UL, _FT_UL, _XB_UL, _XT_UL, e, rr, tr, om)[0]
    d = op_smooth_union(d, _arm(p, _FB_LL, _FT_LL, _XB_LL, _XT_LL, e, rr, tr, om)[0], 0.16)
    d = op_smooth_union(d, _arm(p, _FB_LR, _FT_LR, _XB_LR, _XT_LR, e, rr, tr, om)[0], 0.16)
    d = op_smooth_union(d, _arm(p, _FB_UR, _FT_UR, _XB_UR, _XT_UR, e, rr, tr, om)[0], 0.16)
    d = op_smooth_union(d, sd_sphere(p, 0.20 + 0.16 * e), 0.22)   # centromere pinch
    return d


@wp.func
def _armx(p: wp.vec3, e: float) -> float:
    """the source-card-x of the nearest arm at p (for token colour along the strand)."""
    a0 = _arm(p, _FB_UL, _FT_UL, _XB_UL, _XT_UL, e, 0.2, 0.1, 5.0)
    a1 = _arm(p, _FB_LL, _FT_LL, _XB_LL, _XT_LL, e, 0.2, 0.1, 5.0)
    a2 = _arm(p, _FB_LR, _FT_LR, _XB_LR, _XT_LR, e, 0.2, 0.1, 5.0)
    a3 = _arm(p, _FB_UR, _FT_UR, _XB_UR, _XT_UR, e, 0.2, 0.1, 5.0)
    best = a0[0]; cx = a0[1]
    if a1[0] < best:
        best = a1[0]; cx = a1[1]
    if a2[0] < best:
        best = a2[0]; cx = a2[1]
    if a3[0] < best:
        best = a3[0]; cx = a3[1]
    return cx


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


@wp.func
def _shape(p: wp.vec3, e: float, cerode: float, cmat: float) -> float:
    """union of two proper SDFs: the card ERODES away (cerode grows) while the double-helix
    chromosome MATERIALIZES (cmat shrinks) and coils from the extended DNA (e=0) into the X (e=1).
    A union of valid SDFs morphs cleanly — no fragmentation from lerping distances."""
    dcard = board_map(p) + cerode                   # +cerode erodes the card inward until it vanishes
    dchr = _chromo(p, e) + cmat                      # +cmat hides the strand until it materializes
    return wp.min(dcard, dchr)


@wp.func
def _map(p: wp.vec3, e: float, cerode: float, cmat: float, cagew: float) -> float:
    d = _shape(p, e, cerode, cmat)
    if cagew > 0.01:
        d = wp.min(d, _boxframe(p, _BOXB, 0.02))
    return d


@wp.func
def _normal(p: wp.vec3, e: float, cerode: float, cmat: float, cagew: float) -> wp.vec3:
    ep = 0.0013
    dx = _map(p + wp.vec3(ep, 0.0, 0.0), e, cerode, cmat, cagew) - _map(p - wp.vec3(ep, 0.0, 0.0), e, cerode, cmat, cagew)
    dy = _map(p + wp.vec3(0.0, ep, 0.0), e, cerode, cmat, cagew) - _map(p - wp.vec3(0.0, ep, 0.0), e, cerode, cmat, cagew)
    dz = _map(p + wp.vec3(0.0, 0.0, ep), e, cerode, cmat, cagew) - _map(p - wp.vec3(0.0, 0.0, ep), e, cerode, cmat, cagew)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, e: float, cerode: float, cmat: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _shape(p + n * hr, e, cerode, cmat)
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
def _tok_at_x(cardx: float, index: wp.array3d(dtype=wp.int32), nbx: int, nby: int, nbz: int) -> int:
    fx = (cardx - (-3.7)) / 7.4
    bi = int(wp.clamp(fx * float(nbx), 0.0, float(nbx - 1)))
    by = int(0.5 * float(nby))
    bk = int(0.5 * float(nbz))
    return index[bi, by, bk]


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), index: wp.array3d(dtype=wp.int32),
                   nbx: int, nby: int, nbz: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, e: float, cerode: float, cmat: float,
                   cagew: float, scanx: float, tokamt: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(240):
        p = eye + rd * t
        d = _map(p, e, cerode, cmat, cagew)
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
    # the wire storage-cube
    if cagew > 0.01 and _boxframe(p, _BOXB, 0.02) < _shape(p, e, cerode, cmat) + 0.0005:
        img[i, j] = wp.vec3(0.35, 0.85, 1.0) * (0.8 * cagew)
        return

    n = _normal(p, e, cerode, cmat, cagew)
    ao = _ao(p, n, e, cerode, cmat)
    # token colour: on the strand from its source arm; on the surviving card from world-x
    on_chr = float(0.0)
    if _chromo(p, e) + cmat < board_map(p) + cerode:
        on_chr = 1.0
    cardx = p[0] * (1.0 - on_chr) + _armx(p, e) * on_chr
    tid = _tok_at_x(cardx, index, nbx, nby, nbz)
    tc = _tokcolor(tid)

    lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.5))), 0.2, 1.0)
    base = tc * ((0.42 + 0.62 * lit) * ao)
    # near-transparent scan wave sweeping across the card early on (fades as tokens take over)
    band = wp.abs(p[0] - scanx)
    scan = float(0.0)
    if band < 0.28:
        scan = (1.0 - band / 0.28) * (1.0 - tokamt)
    col = base * tokamt + wp.vec3(0.11, 0.15, 0.14) * (1.0 - tokamt)   # tokens fade in as the scan reads
    col = col + wp.vec3(0.4, 0.9, 1.0) * (scan * 0.45)
    seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
    col = col + tc * (seam * e * 0.5)                     # bright rim as it tightens into the dense X
    img[i, j] = col


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _fwd(s):
    """forward half of the process, s in [0,1]: card -> scan -> DNA strand -> condense to X + cube.
    returns (e, cerode, cmat, cage, scan_x, tok_amt)."""
    if s < 0.22:                                    # the real card + near-transparent scan colours tokens in
        f = _smooth(s / 0.22)
        return 0.0, 0.0, 2.0, 0.0, -3.7 + 7.4 * f, f
    if s < 0.42:                                    # the card erodes away as the DNA strand materialises
        f = _smooth((s - 0.22) / 0.20)
        return 0.0, 2.0 * f, 2.0 * (1.0 - f), 0.0, 4.0, 1.0
    if s < 0.82:                                    # the strand keeps coiling + interlocking into the X
        f = _smooth((s - 0.42) / 0.40)
        return f, 2.0, 0.0, 0.0, 4.0, 1.0
    f = _smooth((s - 0.82) / 0.18)                  # the wire storage-cube draws itself around the dense X
    return 1.0, 2.0, 0.0, f, 4.0, 1.0


def _state(time):
    """whole cycle = forward then exact reverse (triangle over `s`) — the full process, never halfway."""
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    e, cerode, cmat, cagew, scanx, tokamt = _state(time)
    index = wp.array3d(_INDEX, dtype=wp.int32, device=device)

    az = 0.30 + 0.5 * e + 0.08 * math.sin(time * 0.2) + float(mouse[0]) * 0.006
    el = 0.34 * (1.0 - e) + 0.16 * e
    dist = 9.4 * (1.0 - e) + 6.4 * e
    tgt = wp.vec3(0.0, 0.15 + 0.10 * e, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, index, _NBX, _NBY, _NBZ, eye, fwd, right, up, width, height,
                      float(time), tanfov, float(e), float(cerode), float(cmat),
                      float(cagew), float(scanx), float(tokamt)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.15, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3 as the real biology of packing information: a near-transparent scan gives every element "
                "one warp_compress token colour, the tokens thread a DNA double helix laid along the card, "
                "then the strand keeps coiling and interlocking — winding tighter and fatter — until it "
                "condenses into a real metaphase chromosome (the two-arm/two-leg X) of super density, a wire "
                "storage-cube around it; then it unwinds all the way back to the flat card.",
    renderer=_render,
)
