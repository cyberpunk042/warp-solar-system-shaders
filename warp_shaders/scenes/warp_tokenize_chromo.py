"""warp_tokenize_chromo — C3, step 1: TURN THE GRAPHICS CARD INTO TOKENS.

Operator directive (verbatim, 2026-07-16): *"we will do the chromosome by step.... first the only thing,
TURN THE GRAPHIC CARD INTO TOKENS.... TOKENS.... TOKENS."*

So C3 is built **one step at a time**. This is **step 1** and only step 1: the real `gpu_board` becomes
**tokens** — every element replaced by **one coloured cube = its `warp_compress` token** (identical
elements share a colour, exactly like the merge, C1). No DNA, no chromosome yet; those are later steps.

  1. **The card** (the real RTX board).
  2. **A near-transparent scan** sweeps it, reading every element.
  3. **Tokenize** — the card **turns into tokens**: in the scan's wake the board **erodes away** and in its
     place stands a field of **token cubes** — one per element, coloured by its token — a mosaic of the card
     as tokens. When the tokenization is done, the card is gone: it *is* the tokens now.
  4. **Reverse** — the tokens turn back into the card. `time` runs the whole cycle, then loops.

Codec behind the tokens: `warp_compress` block-dedup (the same tokens the merge uses).
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import sd_box
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card

_MAXD = 40.0
_CYCLE = 12.0
_BLOCK = 5
_CY = 0.10                        # token-cube centre height (sits in the card's slab)
_CH = 0.34                        # token-cube height
_FILL = 0.46                      # cube half-width = cell * _FILL (leaves thin seams -> discrete tokens)


def _build():
    """per-column token + occupancy over the card footprint — one token/colour per element."""
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)
    occp = mc._pad_to(occ, b)
    nbx, nby, nbz = index.shape
    blk = (occp.reshape(nbx, b, nby, b, nbz, b)
               .transpose(0, 2, 4, 1, 3, 5)
               .reshape(nbx, nby, nbz, b ** 3))
    occ_blocks = blk.any(axis=3)                        # (nbx, nby, nbz)
    occ_col = occ_blocks.any(axis=1).astype(np.int32)   # (nbx, nbz) occupied column
    tok_col = np.full((nbx, nbz), -1, np.int32)
    for by in range(nby - 1, -1, -1):
        has = occ_blocks[:, by, :]
        f = (tok_col < 0) & has
        tok_col[f] = index[:, by, :][f]                 # topmost occupied block's token
    return np.ascontiguousarray(occ_col), np.ascontiguousarray(tok_col), nbx, nbz


_OCC, _TOK, _NBX, _NBZ = _build()
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
def _tokgrid(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int,
             bwx: float, bwz: float, tmat: float) -> float:
    """SDF of the token-cube field: one cube per occupied column, checked over a 3x3 neighbourhood."""
    ci = int(wp.floor((p[0] + 3.7) / bwx))
    ck = int(wp.floor((p[2] + 1.5) / bwz))
    best = _MAXD
    for di in range(-1, 2):
        for dk in range(-1, 2):
            bi = ci + di
            bk = ck + dk
            if bi >= 0 and bi < nbx and bk >= 0 and bk < nbz:
                if occ[bi, bk] > 0:
                    cx = -3.7 + (float(bi) + 0.5) * bwx
                    cz = -1.5 + (float(bk) + 0.5) * bwz
                    d = sd_box(p - wp.vec3(cx, _CY, cz), wp.vec3(bwx * _FILL, _CH * 0.5, bwz * _FILL))
                    if d < best:
                        best = d
    return best + tmat


@wp.func
def _tokid(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), tok: wp.array2d(dtype=wp.int32),
           nbx: int, nbz: int, bwx: float, bwz: float) -> int:
    """token id of the nearest occupied token-cube to p."""
    ci = int(wp.floor((p[0] + 3.7) / bwx))
    ck = int(wp.floor((p[2] + 1.5) / bwz))
    best = _MAXD
    tid = int(-1)
    for di in range(-1, 2):
        for dk in range(-1, 2):
            bi = ci + di
            bk = ck + dk
            if bi >= 0 and bi < nbx and bk >= 0 and bk < nbz:
                if occ[bi, bk] > 0:
                    cx = -3.7 + (float(bi) + 0.5) * bwx
                    cz = -1.5 + (float(bk) + 0.5) * bwz
                    d = sd_box(p - wp.vec3(cx, _CY, cz), wp.vec3(bwx * _FILL, _CH * 0.5, bwz * _FILL))
                    if d < best:
                        best = d
                        tid = tok[bi, bk]
    return tid


@wp.func
def _shape(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int,
           bwx: float, bwz: float, cerode: float, tmat: float) -> float:
    """the card ERODES away (cerode grows) as the token cubes MATERIALISE (tmat shrinks) — a clean union."""
    return wp.min(board_map(p) + cerode, _tokgrid(p, occ, nbx, nbz, bwx, bwz, tmat))


@wp.func
def _normal(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int,
            bwx: float, bwz: float, cerode: float, tmat: float) -> wp.vec3:
    e = 0.0013
    dx = _shape(p + wp.vec3(e, 0.0, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat) - _shape(p - wp.vec3(e, 0.0, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat)
    dy = _shape(p + wp.vec3(0.0, e, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat) - _shape(p - wp.vec3(0.0, e, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat)
    dz = _shape(p + wp.vec3(0.0, 0.0, e), occ, nbx, nbz, bwx, bwz, cerode, tmat) - _shape(p - wp.vec3(0.0, 0.0, e), occ, nbx, nbz, bwx, bwz, cerode, tmat)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int,
        bwx: float, bwz: float, cerode: float, tmat: float) -> float:
    o = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _shape(p + n * hr, occ, nbx, nbz, bwx, bwz, cerode, tmat)
        o += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * o, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), occ: wp.array2d(dtype=wp.int32),
                   tok: wp.array2d(dtype=wp.int32), nbx: int, nbz: int, bwx: float, bwz: float,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, cerode: float, tmat: float, scanx: float, tokamt: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(200):
        p = eye + rd * t
        d = _shape(p, occ, nbx, nbz, bwx, bwz, cerode, tmat)
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
    n = _normal(p, occ, nbx, nbz, bwx, bwz, cerode, tmat)
    ao = _ao(p, n, occ, nbx, nbz, bwx, bwz, cerode, tmat)
    dcard = board_map(p) + cerode
    dtok = _tokgrid(p, occ, nbx, nbz, bwx, bwz, tmat)

    if dtok < dcard:
        # a token cube: one colour = its element's token
        tid = _tokid(p, occ, tok, nbx, nbz, bwx, bwz)
        tc = _tokcolor(tid)
        lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.5))), 0.2, 1.0)
        img[i, j] = tc * ((0.5 + 0.6 * lit) * ao)
        return

    # the surviving card — real board colours, tinted to its token as the scan reads it
    col = board_shade(p, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)
    if p[0] < scanx:
        tid = _tokid(p, occ, tok, nbx, nbz, bwx, bwz)
        tc = _tokcolor(tid)
        rev = wp.clamp((scanx - p[0]) * 2.5, 0.0, 1.0) * tokamt
        col = col * (1.0 - 0.7 * rev * face) + tc * (0.9 * rev * face)
    band = wp.abs(p[0] - scanx)                              # near-transparent scan wave
    if band < 0.22:
        g = 1.0 - band / 0.22
        col = col + wp.vec3(0.35, 0.8, 1.0) * (g * g * 0.9 * face)
    img[i, j] = col


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _fwd(s):
    """forward half, s in [0,1]: card -> scan -> the card turns into token cubes. (cerode, tmat, scan_x, tok)."""
    if s < 0.40:                                    # near-transparent scan reads the card, tokens preview
        f = _smooth(s / 0.40)
        return 0.0, 2.0, -3.7 + 7.4 * f, f
    f = _smooth((s - 0.40) / 0.60)                  # the card erodes away as the token cubes materialise
    return 2.0 * f, 2.0 * (1.0 - f), 4.0, 1.0


def _state(time):
    """whole cycle = forward then exact reverse (the card turns to tokens and back — never halfway)."""
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    cerode, tmat, scanx, tokamt = _state(time)
    occ = wp.array2d(_OCC, dtype=wp.int32, device=device)
    tok = wp.array2d(_TOK, dtype=wp.int32, device=device)

    az = 0.62 + 0.05 * math.sin(time * 0.15) + float(mouse[0]) * 0.006
    el = 0.55
    dist = 9.4
    tgt = wp.vec3(-0.1, 0.15, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, occ, tok, _NBX, _NBZ, float(_BWX), float(_BWZ),
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(cerode), float(tmat), float(scanx), float(tokamt)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3 step 1 — turn the graphics card into tokens: a near-transparent scan sweeps the real "
                "RTX board and, in its wake, the card erodes away and is replaced by a field of token cubes "
                "(one coloured cube per element = its warp_compress token, identical elements sharing a "
                "colour), then turns back into the card. Later steps weave these tokens into the chromosome.",
    renderer=_render,
)
