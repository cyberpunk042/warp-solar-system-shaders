"""warp_tokenize_chromo — C3: the card is read as a web of token-words, then its REAL MATERIAL coils to a chromosome.

Operator spec (verbatim): *"break down of the item into a web of word that remesent per each atom a word,
or a token rather in a web that gives values and we can then compress it from DNA equivalent sequence
into the whole process of chromosome."* And the physics rule (2026-07-16): *"you have to merge where the
card is ... you are not supposed to break physics"* — the compression must happen to the **real card, in
place**, not as abstract dots teleporting into empty space.

So C3, done the same honest way as the fold (C2):

  1. **The card** — the real RTX `gpu_board` on the bench.
  2. **The web of words** — every element lights up as a glowing **token-node** on the card (its colour =
     its `warp_compress` token/value), linked to its neighbours — the item read as a web of words.
  3. **The DNA sequence** — the web is read out as a **DNA double helix** rising just off the board (the
     token beads become base-pairs on two backbones) — the DNA-equivalent sequence of the card.
  4. **The chromosome** — then the **card's own board material coils up**, in place, into the four arms of
     a metaphase **chromosome** (the classic X, filled with the real green solder-mask / gold-routing /
     GDDR7 / die material — `board_map` wrapped through the coil), exactly as DNA condenses into chromatin.
     The token/DNA read-out fades as the real material takes over. The coiled genome *is* the compressed
     card. `warp_compress.tokenchromo` is the codec behind it (lossless, verified).

`time` runs card → web → DNA → real-material chromosome, then unwinds and loops.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..particles import emitter
from ..procedural.sdf import op_smooth_union, sd_capsule, sd_round_box, sd_sphere
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card, _BB


_MAXD = 40.0
_CYCLE = 15.0
_BLOCK = 5


def _build():
    """token nodes (word per element) + board / DNA-helix positions + web & DNA edges."""
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)
    occp = mc._pad_to(occ, b)
    nbx, nby, nbz = index.shape
    blk = (occp.reshape(nbx, b, nby, b, nbz, b)
               .transpose(0, 2, 4, 1, 3, 5)
               .reshape(nbx, nby, nbz, b ** 3))
    occ_blocks = blk.any(axis=3)
    cells = []
    for bi in range(0, nbx, 1):
        for bk in range(0, nbz, 1):
            tid = -1
            for by in range(nby - 1, -1, -1):
                if occ_blocks[bi, by, bk]:
                    tid = int(index[bi, by, bk]); break
            if tid >= 0:
                cells.append((bi, bk, tid))
    cells = cells[::5]                                     # subsample -> a few dozen legible nodes
    n = len(cells)
    bx, bz = _BB[1], _BB[5]
    board = np.zeros((n, 3), np.float32)
    tok = np.zeros(n, np.int32)
    for i, (bi, bk, tid) in enumerate(cells):
        x = -bx + (bi + 0.5) / nbx * 2.0 * bx
        z = -bz + (bk + 0.5) / nbz * 2.0 * bz
        board[i] = (x, 0.42, z)                            # just above the board's components
        tok[i] = tid

    # DNA double helix rising off the board, standing vertical over the card centre
    helix = np.zeros((n, 3), np.float32)
    npair = max(1, (n + 1) // 2)
    for i in range(n):
        s = i % 2
        k = i // 2
        ay = 0.4 + 3.4 * (k / max(1, npair - 1))           # rises up off the board
        th = k * 0.58
        r = 0.62
        helix[i] = (r * math.cos(th + s * math.pi), ay, r * math.sin(th + s * math.pi))

    web = []
    for i in range(n):
        d = np.sum((board - board[i]) ** 2, axis=1)
        d[i] = 1e9
        for j in np.argsort(d)[:2]:
            a, c = min(i, int(j)), max(i, int(j))
            web.append((a, c))
    web = sorted(set(web))
    dna = []
    for i in range(n - 2):
        dna.append((i, i + 2))                             # backbone (same strand)
    for k in range(npair):
        if 2 * k + 1 < n:
            dna.append((2 * k, 2 * k + 1))                 # base-pair rungs
    web = np.asarray(web, np.int32) if web else np.zeros((0, 2), np.int32)
    dna = np.asarray(dna, np.int32) if dna else np.zeros((0, 2), np.int32)
    nbb = max(0, n - 2)
    return board, helix, tok, web, dna, nbb


_BOARD, _HELIX, _TOK, _WEB, _DNA, _NBB = _build()
_N = _BOARD.shape[0]
_NWEB = _WEB.shape[0]
_NDNA = _DNA.shape[0]


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
    """chromosome-frame point -> board-local coord: stand the flat board up + accordion-pack the arms."""
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
def _cnormal(p: wp.vec3, fold: float) -> wp.vec3:
    e = 0.0013
    dx = _cfmap(p + wp.vec3(e, 0.0, 0.0), fold) - _cfmap(p - wp.vec3(e, 0.0, 0.0), fold)
    dy = _cfmap(p + wp.vec3(0.0, e, 0.0), fold) - _cfmap(p - wp.vec3(0.0, e, 0.0), fold)
    dz = _cfmap(p + wp.vec3(0.0, 0.0, e), fold) - _cfmap(p - wp.vec3(0.0, 0.0, e), fold)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _cao(p: wp.vec3, n: wp.vec3, fold: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _cfmap(p + n * hr, fold)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


# ------------------------------------------------------------------ token web / DNA (particle read-out)
@wp.func
def _hue(h: float) -> wp.vec3:
    r = wp.clamp(wp.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = wp.clamp(2.0 - wp.abs(h * 6.0 - 2.0), 0.0, 1.0)
    bb = wp.clamp(2.0 - wp.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return wp.vec3(r, g, bb)


@wp.func
def _tokcolor(tid: int) -> wp.vec3:
    h = (float(tid) * 0.61803) % 1.0
    return _hue(h)


@wp.func
def _npos(i: int, blend: float, board: wp.array(dtype=wp.vec3), helix: wp.array(dtype=wp.vec3)) -> wp.vec3:
    return board[i] * (1.0 - blend) + helix[i] * blend      # board (web) -> helix (DNA)


@wp.func
def _seg_glow(ro: wp.vec3, rd: wp.vec3, a: wp.vec3, bpt: wp.vec3, size: float) -> float:
    g = float(0.0)
    for s in range(4):
        u = (float(s) + 0.5) / 4.0
        g += emitter(ro, rd, a + (bpt - a) * u, size)
    return g / 4.0


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3),
                   board: wp.array(dtype=wp.vec3), helix: wp.array(dtype=wp.vec3),
                   tok: wp.array(dtype=wp.int32),
                   web: wp.array2d(dtype=wp.int32), dna: wp.array2d(dtype=wp.int32),
                   nnode: int, nweb: int, ndna: int, nbb: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, fold: float, blend: float,
                   node_a: float, web_w: float, dna_w: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    # the real card material — flat board, coiling into the chromosome as `fold` ramps
    t = float(0.0)
    hit = int(0)
    for _ in range(240):
        p = eye + rd * t
        d = _cfmap(p, fold)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.7
        if t > _MAXD:
            break

    col = ec.studio_sky(rd)
    if hit == 1:
        p = eye + rd * t
        n = _cnormal(p, fold)
        ao = _cao(p, n, fold)
        col = board_shade(_fill(p, fold), n, rd, ao, time)
        seam = wp.pow(wp.clamp(1.0 - wp.abs(wp.dot(n, -rd)), 0.0, 1.0), 3.0)
        col = col + wp.vec3(0.3, 0.7, 1.0) * (seam * fold * 0.6)     # cool rim glow as it coils

    # the token/DNA read-out (glowing web -> helix), fading as the material coils
    glow = wp.vec3(0.0, 0.0, 0.0)
    for k in range(nnode):
        pk = _npos(k, blend, board, helix)
        ge = emitter(eye, rd, pk, 0.07)
        glow = glow + _tokcolor(tok[k]) * (ge * node_a * 1.4)
    if web_w > 0.01:
        for e in range(nweb):
            a = _npos(web[e, 0], blend, board, helix)
            bp = _npos(web[e, 1], blend, board, helix)
            g = _seg_glow(eye, rd, a, bp, 0.05)
            glow = glow + wp.vec3(0.6, 0.85, 1.0) * (g * web_w * 0.7)
    if dna_w > 0.01:
        for e in range(ndna):
            a = _npos(dna[e, 0], blend, board, helix)
            bp = _npos(dna[e, 1], blend, board, helix)
            if e < nbb:
                g = _seg_glow(eye, rd, a, bp, 0.032)
                glow = glow + wp.vec3(0.55, 0.9, 1.0) * (g * dna_w * 1.5)
            else:
                g = _seg_glow(eye, rd, a, bp, 0.05)
                tc = (_tokcolor(tok[dna[e, 0]]) + _tokcolor(tok[dna[e, 1]])) * 0.5
                glow = glow + tc * (g * dna_w * 0.9)

    img[i, j] = col + glow


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _stage(time):
    """(fold, blend, node_alpha, web_w, dna_w) — card → web → DNA → real-material chromosome → unwind."""
    u = (float(time) % _CYCLE) / _CYCLE
    if u < 0.12:                                    # the card (flat, real material)
        return 0.0, 0.0, _smooth(u / 0.12) * 0.2, 0.0, 0.0
    if u < 0.32:                                    # break into the web of token-words (on the card)
        f = (u - 0.12) / 0.20
        return 0.0, 0.0, _smooth(f), _smooth(f), 0.0
    if u < 0.52:                                    # read out into the DNA helix (rising off the card)
        f = (u - 0.32) / 0.20
        return 0.0, _smooth(f), 1.0, 1.0 - _smooth(f), _smooth(f)
    if u < 0.82:                                    # the card's MATERIAL coils into the chromosome
        f = (u - 0.52) / 0.30
        return _smooth(f), 1.0, 1.0 - _smooth(f), 0.0, 1.0 - _smooth(f)
    # unwind: the chromosome opens back to the flat card, DNA re-reads
    f = (u - 0.82) / 0.18
    return 1.0 - _smooth(f), 1.0 - _smooth(f), _smooth(f), 0.0, _smooth(f)


def _render(width, height, time, mouse, device):
    fold, blend, node_a, web_w, dna_w = _stage(time)
    board = wp.array(_BOARD, dtype=wp.vec3, device=device)
    helix = wp.array(_HELIX, dtype=wp.vec3, device=device)
    tok = wp.array(_TOK, dtype=wp.int32, device=device)
    web = wp.array2d(_WEB, dtype=wp.int32, device=device)
    dna = wp.array2d(_DNA, dtype=wp.int32, device=device)

    az = 0.30 + 0.35 * fold + 0.10 * math.sin(time * 0.2) + float(mouse[0]) * 0.006
    el = 0.30 * (1.0 - fold) + 0.16 * fold                  # look down at the board, rise to face the X
    dist = 9.4 * (1.0 - fold) + 6.2 * fold
    tgt = wp.vec3(0.0, 0.2 + 0.2 * fold, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, board, helix, tok, web, dna, _N, _NWEB, _NDNA, _NBB,
                      eye, fwd, right, up, width, height, float(time), tanfov,
                      float(fold), float(blend), float(node_a), float(web_w), float(dna_w)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.15, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3 as a physically-honest process: the real RTX board is read as a web of token-words "
                "(each element a glowing node = its warp_compress token/value), the web reads out as a DNA "
                "double helix, then the card's OWN board material coils in place into the four arms of a "
                "metaphase chromosome (the real green mask / gold routing / GDDR7 / die, wrapped through "
                "the coil) — the coiled genome is the compressed card (tokenchromo codec, lossless).",
    renderer=_render,
)
