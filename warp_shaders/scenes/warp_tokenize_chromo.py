"""warp_tokenize_chromo — C3, step by step: card -> TOKENS -> tokens CONNECT (proximity) into a strand.

Operator directives (verbatim):
  - step 1: *"first the only thing, TURN THE GRAPHIC CARD INTO TOKENS.... TOKENS.... TOKENS."*
  - step 2: *"now as they become token, they now connect with the proximity token(s) to form a DNA strain,
    continuous strain that will be able to weave itself like this later. you start from the base pair
    merging and then it turn into a double helix and tighter and tighter till you can center it at the
    telomere into the chromosome."*
  - density: *"there is also so much more token in a card... like at least a million. do it right."*

THE LAW I kept breaking (and now honour): do NOT fabricate a shape and teleport the card into it. Grow the
structure from the REAL tokens, in place, by adjacency. The strand is not a canned helix dropped on an axis
— it is the card's own tokens **connected to their proximity neighbours, where the card is**, into one
**continuous** thread. (That thread is what later coils and weaves into the chromosome — next steps.)

  1. **The card** (real RTX board).
  2. **Scan -> tokens** (step 1): the card erodes into a dense field of token cubes (one colour per element).
  3. **Connect** (step 2): each token **links to its neighbour** — a link front sweeps along the card and
     the tokens join, neighbour to neighbour, into ONE continuous strand threaded through where they sit.
  4. **Reverse** — the strand unlinks back to loose tokens and the card. `time` runs the cycle, then loops.

Later steps coil this in-place strand into the DNA double helix, nucleosomes, and the chromosome X.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import sd_box, sd_capsule
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card

_MAXD = 40.0
_CYCLE = 14.0
_BLOCK = 2                        # dense tokens — a real card has thousands, not dozens
_CY = 0.10                        # token-cube centre height
_CH = 0.34
_FILL = 0.46
_SY = 0.34                        # the strand lifts clearly ABOVE the token field so you SEE it form
_SR = 0.045                       # strand tube radius (links neighbouring tokens)


def _build():
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)
    occp = mc._pad_to(occ, b)
    nbx, nby, nbz = index.shape
    blk = (occp.reshape(nbx, b, nby, b, nbz, b)
               .transpose(0, 2, 4, 1, 3, 5)
               .reshape(nbx, nby, nbz, b ** 3))
    occ_blocks = blk.any(axis=3)
    occ_col = occ_blocks.any(axis=1).astype(np.int32)
    tok_col = np.full((nbx, nbz), -1, np.int32)
    for by in range(nby - 1, -1, -1):
        has = occ_blocks[:, by, :]
        f = (tok_col < 0) & has
        tok_col[f] = index[:, by, :][f]
    # a CONTINUOUS path through the occupied token cells, following spatial ADJACENCY (serpentine): the
    # order in which proximity neighbours link into the strand. per cell: its path index + the offset to
    # its path successor (the neighbour it connects to).
    order = []
    for bk in range(nbz):
        rng = range(nbx) if (bk % 2 == 0) else range(nbx - 1, -1, -1)
        for bi in rng:
            if occ_col[bi, bk] > 0:
                order.append((bi, bk))
    n = len(order)
    pidx = np.full((nbx, nbz), -1, np.int32)
    sdi = np.zeros((nbx, nbz), np.int32)
    sdk = np.zeros((nbx, nbz), np.int32)
    for i, (bi, bk) in enumerate(order):
        pidx[bi, bk] = i
        if i + 1 < n:
            nbi, nbk = order[i + 1]
            sdi[bi, bk] = nbi - bi
            sdk[bi, bk] = nbk - bk
    return (np.ascontiguousarray(occ_col), np.ascontiguousarray(tok_col),
            np.ascontiguousarray(sdi), np.ascontiguousarray(sdk), np.ascontiguousarray(pidx), n, nbx, nbz)


_OCC, _TOK, _SDI, _SDK, _PIDX, _NPATH, _NBX, _NBZ = _build()
_BWX = 7.4 / _NBX
_BWZ = 3.0 / _NBZ


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
def _cellc(bi: int, bk: int, bwx: float, bwz: float, y: float) -> wp.vec3:
    return wp.vec3(-3.7 + (float(bi) + 0.5) * bwx, y, -1.5 + (float(bk) + 0.5) * bwz)


# --------------------------------------------------------------------------- token field (step 1)
@wp.func
def _tokgrid(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), pidx: wp.array2d(dtype=wp.int32),
             nbx: int, nbz: int, bwx: float, bwz: float, grow: float, tmat: float) -> float:
    """the LOOSE tokens — a cube stays in the field only until the link front reaches it (pidx >= grow);
    once linked it has left the field for the strand above."""
    ci = int(wp.floor((p[0] + 3.7) / bwx))
    ck = int(wp.floor((p[2] + 1.5) / bwz))
    best = _MAXD
    for di in range(-1, 2):
        for dk in range(-1, 2):
            bi = ci + di
            bk = ck + dk
            if bi >= 0 and bi < nbx and bk >= 0 and bk < nbz:
                if occ[bi, bk] > 0 and float(pidx[bi, bk]) >= grow:
                    c = _cellc(bi, bk, bwx, bwz, _CY)
                    d = sd_box(p - c, wp.vec3(bwx * _FILL, _CH * 0.5, bwz * _FILL))
                    if d < best:
                        best = d
    return best + tmat


@wp.func
def _tokid(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), tok: wp.array2d(dtype=wp.int32),
           pidx: wp.array2d(dtype=wp.int32), nbx: int, nbz: int, bwx: float, bwz: float, grow: float) -> int:
    ci = int(wp.floor((p[0] + 3.7) / bwx))
    ck = int(wp.floor((p[2] + 1.5) / bwz))
    best = _MAXD
    tid = int(-1)
    for di in range(-1, 2):
        for dk in range(-1, 2):
            bi = ci + di
            bk = ck + dk
            if bi >= 0 and bi < nbx and bk >= 0 and bk < nbz:
                if occ[bi, bk] > 0 and float(pidx[bi, bk]) >= grow:
                    c = _cellc(bi, bk, bwx, bwz, _CY)
                    d = sd_box(p - c, wp.vec3(bwx * _FILL, _CH * 0.5, bwz * _FILL))
                    if d < best:
                        best = d
                        tid = tok[bi, bk]
    return tid


# --------------------------------------------------------- proximity-linked strand (step 2, IN PLACE)
@wp.func
def _strand(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), sdi: wp.array2d(dtype=wp.int32),
            sdk: wp.array2d(dtype=wp.int32), pidx: wp.array2d(dtype=wp.int32),
            nbx: int, nbz: int, bwx: float, bwz: float, grow: float, shide: float) -> float:
    """the strand: each linked token connects to its path-successor (its proximity neighbour). It grows
    as the link front `grow` sweeps along the path. Checked over a 3x3 cell neighbourhood."""
    ci = int(wp.floor((p[0] + 3.7) / bwx))
    ck = int(wp.floor((p[2] + 1.5) / bwz))
    best = _MAXD
    for di in range(-1, 2):
        for dk in range(-1, 2):
            bi = ci + di
            bk = ck + dk
            if bi >= 0 and bi < nbx and bk >= 0 and bk < nbz:
                if occ[bi, bk] > 0 and float(pidx[bi, bk]) < grow:
                    a = _cellc(bi, bk, bwx, bwz, _SY)
                    ei = bi + sdi[bi, bk]
                    ek = bk + sdk[bi, bk]
                    bb = _cellc(ei, ek, bwx, bwz, _SY)          # the neighbour it links to
                    d = sd_capsule(p, a, bb, _SR)
                    if d < best:
                        best = d
    return best + shide


@wp.func
def _strand_tok(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), tok: wp.array2d(dtype=wp.int32),
                sdi: wp.array2d(dtype=wp.int32), sdk: wp.array2d(dtype=wp.int32),
                pidx: wp.array2d(dtype=wp.int32), nbx: int, nbz: int, bwx: float, bwz: float,
                grow: float) -> int:
    ci = int(wp.floor((p[0] + 3.7) / bwx))
    ck = int(wp.floor((p[2] + 1.5) / bwz))
    best = _MAXD
    tid = int(-1)
    for di in range(-1, 2):
        for dk in range(-1, 2):
            bi = ci + di
            bk = ck + dk
            if bi >= 0 and bi < nbx and bk >= 0 and bk < nbz:
                if occ[bi, bk] > 0 and float(pidx[bi, bk]) < grow:
                    a = _cellc(bi, bk, bwx, bwz, _SY)
                    ei = bi + sdi[bi, bk]
                    ek = bk + sdk[bi, bk]
                    bb = _cellc(ei, ek, bwx, bwz, _SY)
                    d = sd_capsule(p, a, bb, _SR)
                    if d < best:
                        best = d
                        tid = tok[bi, bk]
    return tid


# --------------------------------------------------------------------------- combined field
@wp.func
def _shape(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), sdi: wp.array2d(dtype=wp.int32),
           sdk: wp.array2d(dtype=wp.int32), pidx: wp.array2d(dtype=wp.int32),
           nbx: int, nbz: int, bwx: float, bwz: float,
           cerode: float, tmat: float, grow: float, shide: float) -> float:
    dcard = board_map(p) + cerode
    dtok = _tokgrid(p, occ, pidx, nbx, nbz, bwx, bwz, grow, tmat)
    dstr = _strand(p, occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, grow, shide)
    return wp.min(dcard, wp.min(dtok, dstr))


@wp.func
def _normal(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), sdi: wp.array2d(dtype=wp.int32),
            sdk: wp.array2d(dtype=wp.int32), pidx: wp.array2d(dtype=wp.int32),
            nbx: int, nbz: int, bwx: float, bwz: float,
            cerode: float, tmat: float, grow: float, shide: float) -> wp.vec3:
    e = 0.0013
    dx = _shape(p + wp.vec3(e, 0.0, 0.0), occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide) - _shape(p - wp.vec3(e, 0.0, 0.0), occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
    dy = _shape(p + wp.vec3(0.0, e, 0.0), occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide) - _shape(p - wp.vec3(0.0, e, 0.0), occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
    dz = _shape(p + wp.vec3(0.0, 0.0, e), occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide) - _shape(p - wp.vec3(0.0, 0.0, e), occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, occ: wp.array2d(dtype=wp.int32), sdi: wp.array2d(dtype=wp.int32),
        sdk: wp.array2d(dtype=wp.int32), pidx: wp.array2d(dtype=wp.int32),
        nbx: int, nbz: int, bwx: float, bwz: float,
        cerode: float, tmat: float, grow: float, shide: float) -> float:
    o = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _shape(p + n * hr, occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
        o += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * o, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), occ: wp.array2d(dtype=wp.int32),
                   tok: wp.array2d(dtype=wp.int32), sdi: wp.array2d(dtype=wp.int32),
                   sdk: wp.array2d(dtype=wp.int32), pidx: wp.array2d(dtype=wp.int32),
                   nbx: int, nbz: int, bwx: float, bwz: float,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, cerode: float, tmat: float, grow: float, shide: float,
                   scanx: float, tokamt: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(210):
        p = eye + rd * t
        d = _shape(p, occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.75
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
    ao = _ao(p, n, occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, cerode, tmat, grow, shide)
    dcard = board_map(p) + cerode
    dtok = _tokgrid(p, occ, pidx, nbx, nbz, bwx, bwz, grow, tmat)
    dstr = _strand(p, occ, sdi, sdk, pidx, nbx, nbz, bwx, bwz, grow, shide)
    lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.5))), 0.2, 1.0)

    if dstr < dcard and dstr < dtok:
        tc = _tokcolor(_strand_tok(p, occ, tok, sdi, sdk, pidx, nbx, nbz, bwx, bwz, grow))
        img[i, j] = tc * ((0.55 + 0.6 * lit) * ao) + tc * 0.12     # the strand, slightly brighter than field
        return

    if dtok < dcard:
        tc = _tokcolor(_tokid(p, occ, tok, pidx, nbx, nbz, bwx, bwz, grow))
        img[i, j] = tc * ((0.45 + 0.55 * lit) * ao)
        return

    col = board_shade(p, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)
    if p[0] < scanx:
        tc = _tokcolor(_tokid(p, occ, tok, pidx, nbx, nbz, bwx, bwz, grow))
        rev = wp.clamp((scanx - p[0]) * 2.5, 0.0, 1.0) * tokamt
        col = col * (1.0 - 0.7 * rev * face) + tc * (0.9 * rev * face)
    band = wp.abs(p[0] - scanx)
    if band < 0.22:
        g = 1.0 - band / 0.22
        col = col + wp.vec3(0.35, 0.8, 1.0) * (g * g * 0.9 * face)
    img[i, j] = col


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _fwd(s):
    """forward half, s in [0,1]. (cerode, tmat, grow, shide, scan_x, tok_amt)."""
    if s < 0.20:                                    # scan reads the card, tokens preview in
        f = _smooth(s / 0.20)
        return 0.0, 2.0, 0.0, 3.0, -3.7 + 7.4 * f, f
    if s < 0.34:                                    # card erodes away -> the dense field of token cubes
        f = _smooth((s - 0.20) / 0.14)
        return 2.0 * f, 2.0 * (1.0 - f), 0.0, 3.0, 4.0, 1.0
    if s < 0.44:                                    # HOLD: the card is now a dense field of loose tokens
        return 2.0, 0.0, 0.0, 3.0, 4.0, 1.0
    # CONNECT: the link front sweeps the card, tokens join neighbour-to-neighbour into ONE strand, in place
    f = _smooth((s - 0.44) / 0.56)
    return 2.0, 0.0, float(_NPATH + 2) * f, 3.0 * (1.0 - _smooth((s - 0.44) / 0.20)), 4.0, 1.0


def _state(time):
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    cerode, tmat, grow, shide, scanx, tokamt = _state(time)
    occ = wp.array2d(_OCC, dtype=wp.int32, device=device)
    tok = wp.array2d(_TOK, dtype=wp.int32, device=device)
    sdi = wp.array2d(_SDI, dtype=wp.int32, device=device)
    sdk = wp.array2d(_SDK, dtype=wp.int32, device=device)
    pidx = wp.array2d(_PIDX, dtype=wp.int32, device=device)

    az = 0.60 + 0.05 * math.sin(time * 0.15) + float(mouse[0]) * 0.006   # stays on the card — in place
    el = 0.58
    dist = 9.2
    tgt = wp.vec3(-0.1, 0.12, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, occ, tok, sdi, sdk, pidx, _NBX, _NBZ, float(_BWX), float(_BWZ),
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(cerode), float(tmat), float(grow), float(shide),
                      float(scanx), float(tokamt)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3, step by step: a near-transparent scan turns the real RTX board into a dense field of "
                "token cubes (one colour per element), then a link front sweeps the card and the tokens "
                "connect to their proximity neighbours — where they sit, not teleported — into one "
                "continuous strand threaded through the card; then it unlinks back to the card. Later steps "
                "coil this in-place strand into the DNA double helix and the chromosome.",
    renderer=_render,
)
