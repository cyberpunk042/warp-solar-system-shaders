"""warp_tokenize_chromo — C3, built step by step: card -> TOKENS -> DNA double helix (-> chromosome later).

Operator directives (verbatim):
  - 2026-07-16 step 1: *"first the only thing, TURN THE GRAPHIC CARD INTO TOKENS.... TOKENS.... TOKENS."*
  - 2026-07-16 step 2: *"now as they become token, they now connect with the proximity token(s) to form a
    DNA strain, continuous strain ... you start from the base pair merging and then it turn into a double
    helix and tighter and tighter till you can center it at the telomere into the chromosome."*
  - 2026-07-16 density: *"there is also so much more token in a card... like at least a million in this card.
    do it right."*

So C3 follows the real chromosome hierarchy (base pairs -> double helix -> nucleosomes -> chromatid ->
chromosome), one verified step at a time, and the card is DENSE with tokens (a real card has millions):

  1. **The card** (real RTX board).
  2. **Scan -> tokens** (step 1): a near-transparent scan sweeps the board; in its wake the card erodes
     away and is replaced by a **dense field of token cubes** — one colour per element = its `warp_compress`
     token (fine blocks: thousands of cells, not dozens).
  3. **Tokens -> DNA** (step 2, THIS): the tokens **connect** into a continuous strand — the card's whole
     token sequence unrolled into **base-pair rungs** (densely packed, coloured by the element each came
     from) between two backbone rails — a **DNA double helix** that winds **tighter and tighter**.
  4. **Reverse** — it unwinds back through tokens to the card. `time` runs the whole cycle, then loops.

Later steps (3+) coil this helix into nucleosomes and condense it into the metaphase chromosome X.
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
_BLOCK = 2                        # fine blocks -> a real card is DENSE with tokens (thousands, not dozens)
_CY = 0.10
_CH = 0.34
_FILL = 0.46
_R = 0.52                         # DNA backbone radius
_RRAIL = 0.11                     # backbone rail tube radius
_RRUNG = 0.024                    # base-pair rung radius
_NRUNG = 150                      # base pairs shown along the strand — a dense ladder (subsamples the card)
_OMMAX = 2.6                      # max twist (rad/unit) — a clean ~10-base-pair-per-turn double helix
_L = 3.7


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
    # the card's whole token sequence, unrolled (every occupied cell) — this becomes the DNA base pairs
    seq = tok_col[occ_col > 0].astype(np.int32)
    seq = seq[seq >= 0]
    if seq.size == 0:
        seq = np.zeros(1, np.int32)
    return (np.ascontiguousarray(occ_col), np.ascontiguousarray(tok_col),
            np.ascontiguousarray(seq), nbx, nbz)


_OCC, _TOK, _SEQ, _NBX, _NBZ = _build()
_NSEQ = int(_SEQ.shape[0])
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


# --------------------------------------------------------------------------- token field (step 1)
@wp.func
def _tokgrid(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int,
             bwx: float, bwz: float, tmat: float) -> float:
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


# --------------------------------------------------------------------------- DNA double helix (step 2)
@wp.func
def _rail(p: wp.vec3, om: float, phase: float) -> float:
    s = wp.clamp(p[0], -_L, _L)
    py = p[1]
    pz = p[2]
    rad = wp.sqrt(py * py + pz * pz)
    ang = wp.atan2(pz, py)
    dang = ang - (om * s + phase)
    dang = dang - 6.2831853 * wp.round(dang / 6.2831853)
    axo = p[0] - s
    hr = wp.sqrt((rad - _R) * (rad - _R) + (_R * dang) * (_R * dang))
    return wp.sqrt(hr * hr + axo * axo) - _RRAIL


@wp.func
def _rails(p: wp.vec3, om: float) -> float:
    return wp.min(_rail(p, om, 0.0), _rail(p, om, 3.14159265))


@wp.func
def _rungx(p: wp.vec3, m: int) -> float:
    return -_L + (float(m) + 0.5) / float(_NRUNG) * 2.0 * _L


@wp.func
def _rung(p: wp.vec3, om: float) -> float:
    """the base pairs: _NRUNG diameter bars packed along the strand, each linking the two rails."""
    fm = (p[0] + _L) / (2.0 * _L) * float(_NRUNG) - 0.5
    m = int(wp.round(fm))
    d = _MAXD
    if m >= 0 and m < _NRUNG:
        xm = _rungx(p, m)
        a = om * xm
        dir = wp.vec3(0.0, wp.cos(a), wp.sin(a))
        base = wp.vec3(xm, 0.0, 0.0)
        d = sd_capsule(p, base + dir * _R, base - dir * _R, _RRUNG)
    return d


@wp.func
def _rung_tok(p: wp.vec3, seq: wp.array(dtype=wp.int32), nseq: int) -> int:
    fm = (p[0] + _L) / (2.0 * _L) * float(_NRUNG) - 0.5
    m = int(wp.round(fm))
    tid = int(-1)
    if m >= 0 and m < _NRUNG:
        k = int(float(m) * float(nseq) / float(_NRUNG))
        if k >= 0 and k < nseq:
            tid = seq[k]
    return tid


@wp.func
def _dna(p: wp.vec3, om: float, dnahide: float) -> float:
    return wp.min(_rails(p, om), _rung(p, om)) + dnahide


# --------------------------------------------------------------------------- combined field
@wp.func
def _shape(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int, bwx: float, bwz: float,
           cerode: float, tmat: float, dnahide: float, om: float) -> float:
    dcard = board_map(p) + cerode
    dtok = _tokgrid(p, occ, nbx, nbz, bwx, bwz, tmat)
    ddna = _dna(p, om, dnahide)
    return wp.min(dcard, wp.min(dtok, ddna))


@wp.func
def _normal(p: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int, bwx: float, bwz: float,
            cerode: float, tmat: float, dnahide: float, om: float) -> wp.vec3:
    e = 0.0013
    dx = _shape(p + wp.vec3(e, 0.0, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om) - _shape(p - wp.vec3(e, 0.0, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
    dy = _shape(p + wp.vec3(0.0, e, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om) - _shape(p - wp.vec3(0.0, e, 0.0), occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
    dz = _shape(p + wp.vec3(0.0, 0.0, e), occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om) - _shape(p - wp.vec3(0.0, 0.0, e), occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, occ: wp.array2d(dtype=wp.int32), nbx: int, nbz: int, bwx: float, bwz: float,
        cerode: float, tmat: float, dnahide: float, om: float) -> float:
    o = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _shape(p + n * hr, occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
        o += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * o, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), occ: wp.array2d(dtype=wp.int32),
                   tok: wp.array2d(dtype=wp.int32), seq: wp.array(dtype=wp.int32), nseq: int,
                   nbx: int, nbz: int, bwx: float, bwz: float,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, cerode: float, tmat: float, dnahide: float, om: float,
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
        d = _shape(p, occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.72
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    n = _normal(p, occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
    ao = _ao(p, n, occ, nbx, nbz, bwx, bwz, cerode, tmat, dnahide, om)
    dcard = board_map(p) + cerode
    dtok = _tokgrid(p, occ, nbx, nbz, bwx, bwz, tmat)
    ddna = _dna(p, om, dnahide)
    lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.5))), 0.2, 1.0)

    if ddna < dcard and ddna < dtok:
        drung = _rung(p, om) + dnahide
        if drung < ddna + 0.0006:
            tc = _tokcolor(_rung_tok(p, seq, nseq))          # base pair = one of the card's tokens
            img[i, j] = tc * ((0.5 + 0.6 * lit) * ao)
        else:
            img[i, j] = wp.vec3(0.82, 0.85, 0.9) * ((0.4 + 0.6 * lit) * ao)   # sugar-phosphate backbone
        return

    if dtok < dcard:
        tc = _tokcolor(_tokid(p, occ, tok, nbx, nbz, bwx, bwz))
        img[i, j] = tc * ((0.5 + 0.6 * lit) * ao)
        return

    col = board_shade(p, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)
    if p[0] < scanx:
        tc = _tokcolor(_tokid(p, occ, tok, nbx, nbz, bwx, bwz))
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
    """forward half, s in [0,1]. (cerode, tmat, dnahide, om, scan_x, tok_amt)."""
    if s < 0.20:                                    # scan reads the card, tokens preview in
        f = _smooth(s / 0.20)
        return 0.0, 2.0, 3.0, 0.0, -3.7 + 7.4 * f, f
    if s < 0.34:                                    # card erodes away -> the dense field of token cubes
        f = _smooth((s - 0.20) / 0.14)
        return 2.0 * f, 2.0 * (1.0 - f), 3.0, 0.0, 4.0, 1.0
    if s < 0.46:                                    # HOLD: the card is now a dense field of tokens
        return 2.0, 0.0, 3.0, 0.0, 4.0, 1.0
    if s < 0.62:                                    # tokens CONNECT -> a base-pair ladder (DNA, untwisted)
        f = _smooth((s - 0.46) / 0.16)
        return 2.0, 2.0 * f, 3.0 * (1.0 - f), 0.0, 4.0, 1.0
    f = _smooth((s - 0.62) / 0.38)                  # the ladder twists into a double helix, tighter + tighter
    return 2.0, 2.0, 0.0, _OMMAX * f, 4.0, 1.0


def _state(time):
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    cerode, tmat, dnahide, om, scanx, tokamt = _state(time)
    occ = wp.array2d(_OCC, dtype=wp.int32, device=device)
    tok = wp.array2d(_TOK, dtype=wp.int32, device=device)
    seq = wp.array(_SEQ, dtype=wp.int32, device=device)

    helixamt = min(1.0, om / _OMMAX)
    az = 0.58 - 0.34 * helixamt + 0.05 * math.sin(time * 0.15) + float(mouse[0]) * 0.006   # swing side-on
    el = 0.55 * (1.0 - helixamt) + 0.22 * helixamt
    dist = 9.4 * (1.0 - helixamt) + 6.8 * helixamt
    tgt = wp.vec3(-0.1, 0.15 * (1.0 - helixamt), 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, occ, tok, seq, _NSEQ, _NBX, _NBZ, float(_BWX), float(_BWZ),
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(cerode), float(tmat), float(dnahide), float(om),
                      float(scanx), float(tokamt)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3, built step by step: a near-transparent scan turns the real RTX board into a dense field "
                "of token cubes (one colour per element), then the card's whole token sequence connects into "
                "a DNA double helix — densely-packed base-pair rungs (each a card token) between two "
                "sugar-phosphate backbones — that winds tighter and tighter; then it unwinds back to the "
                "card. Later steps coil this helix into the chromosome.",
    renderer=_render,
)
