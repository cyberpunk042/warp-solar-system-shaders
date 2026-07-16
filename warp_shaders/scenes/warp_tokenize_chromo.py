"""warp_tokenize_chromo — C3: the card broken into a web of token-words, read as DNA, coiled to a chromosome.

Operator spec (verbatim): *"break down of the item into a web of word that remesent per each atom a word,
or a token rather in a web that gives values and we can then compress it from DNA equivalent sequence
into the whole process of chromosome."*

The tokenize -> chromosome compression (C3) as a watchable process on the real `gpu_board`:

  1. **The card** — the real RTX board sits on the bench.
  2. **The web of words** — it breaks down into a **web of tokens**: every element becomes a glowing node
     (a *word*, its colour = its `warp_compress` token, its value), linked to its neighbours — the item as
     a web of words that give values.
  3. **The DNA sequence** — the web is read out as a **sequence** and snaps into a **DNA double helix**:
     the tokens become coloured base-pairs on two backbones — the *DNA-equivalent sequence*.
  4. **The chromosome** — the helix **coils** through nucleosome-like packing into the four arms of a
     metaphase **chromosome** (the X) — *the whole process of chromosome*. The coiled genome is the
     compressed card; `warp_compress.tokenchromo` is the codec behind it (lossless, verified).

`time` runs card -> web -> DNA -> chromosome, then unwinds and loops.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..particles import emitter
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card, _BB


_MAXD = 40.0
_CYCLE = 14.0
_BLOCK = 5


def _build():
    """token nodes (word per element) + their board / helix / chromosome positions + web & DNA edges."""
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)
    occp = mc._pad_to(occ, b)
    nbx, nby, nbz = index.shape
    blk = (occp.reshape(nbx, b, nby, b, nbz, b)
               .transpose(0, 2, 4, 1, 3, 5)
               .reshape(nbx, nby, nbz, b ** 3))
    occ_blocks = blk.any(axis=3)
    # one token-word per occupied board column (top-most block), subsampled to a legible node count
    cells = []
    for bi in range(0, nbx, 1):
        for bk in range(0, nbz, 1):
            tid = -1
            for by in range(nby - 1, -1, -1):
                if occ_blocks[bi, by, bk]:
                    tid = int(index[bi, by, bk]); break
            if tid >= 0:
                cells.append((bi, bk, tid))
    cells = cells[::5]                                     # subsample -> ~a few dozen legible nodes
    n = len(cells)
    bx, bz = _BB[1], _BB[5]
    board = np.zeros((n, 3), np.float32)
    tok = np.zeros(n, np.int32)
    for i, (bi, bk, tid) in enumerate(cells):
        x = -bx + (bi + 0.5) / nbx * 2.0 * bx
        z = -bz + (bk + 0.5) / nbz * 2.0 * bz
        board[i] = (x, 0.22, z)
        tok[i] = tid

    # DNA double helix: node i on strand i%2, backbone step k=i//2, standing VERTICAL (along y) so it
    # reads instantly as DNA; the two strands are pi out of phase, base-pair rungs bridge them
    helix = np.zeros((n, 3), np.float32)
    npair = max(1, (n + 1) // 2)
    for i in range(n):
        s = i % 2
        k = i // 2
        ay = -2.35 + 4.7 * (k / max(1, npair - 1))
        th = k * 0.58                                    # ~11 steps/turn -> clear open coil
        r = 0.72
        helix[i] = (r * math.cos(th + s * math.pi), ay, r * math.sin(th + s * math.pi))

    # chromosome X: four arms from a pinched centromere, the sequence coiled along them
    chromo = np.zeros((n, 3), np.float32)
    dirs = [(0.6, 0.85), (-0.6, 0.85), (0.6, -0.85), (-0.6, -0.85)]
    for i in range(n):
        s01 = i / max(1, n - 1)
        arm = min(3, int(s01 * 4))
        tt = s01 * 4.0 - arm
        dx, dy = dirs[arm]
        a0, L = 0.30, 1.5
        rad = a0 + L * tt
        # a little helical coil around the arm axis -> packed nucleosome look
        coil = 0.18
        cang = i * 1.1
        px = dx * rad + math.cos(cang) * coil * (-dy)
        py = dy * rad + math.cos(cang) * coil * (dx)
        pz = math.sin(cang) * coil
        chromo[i] = (px, py, pz)

    # web edges: link each node to its 2 nearest neighbours in board space
    web = []
    for i in range(n):
        d = np.sum((board - board[i]) ** 2, axis=1)
        d[i] = 1e9
        for j in np.argsort(d)[:2]:
            a, c = min(i, int(j)), max(i, int(j))
            web.append((a, c))
    web = sorted(set(web))
    # dna edges: backbone (i -> i+2, same strand) + base-pair rungs (2k -> 2k+1)
    dna = []
    for i in range(n - 2):
        dna.append((i, i + 2))
    for k in range(npair):
        if 2 * k + 1 < n:
            dna.append((2 * k, 2 * k + 1))
    web = np.asarray(web, np.int32) if web else np.zeros((0, 2), np.int32)
    dna = np.asarray(dna, np.int32) if dna else np.zeros((0, 2), np.int32)
    return board, helix, chromo, tok, web, dna


_BOARD, _HELIX, _CHROMO, _TOK, _WEB, _DNA = _build()
_N = _BOARD.shape[0]
_NWEB = _WEB.shape[0]
_NDNA = _DNA.shape[0]
_NBB = max(0, _N - 2)            # first (_N-2) DNA edges are backbone (i, i+2); the rest are rungs


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
def _npos(i: int, stage: int, blend: float,
          board: wp.array(dtype=wp.vec3), helix: wp.array(dtype=wp.vec3),
          chromo: wp.array(dtype=wp.vec3)) -> wp.vec3:
    p = board[i]                                           # card / web
    if stage == 1:
        p = board[i] * (1.0 - blend) + helix[i] * blend    # -> DNA helix
    if stage >= 2:
        p = helix[i] * (1.0 - blend) + chromo[i] * blend   # -> chromosome X
    return p


@wp.func
def _seg_glow(ro: wp.vec3, rd: wp.vec3, a: wp.vec3, bpt: wp.vec3, size: float) -> float:
    g = float(0.0)
    for s in range(4):
        u = (float(s) + 0.5) / 4.0
        g += emitter(ro, rd, a + (bpt - a) * u, size)
    return g / 4.0


@wp.func
def _spin(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.10 * wp.sin(time * 0.16)
    ca = wp.cos(a); sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _cmap(p: wp.vec3, time: float) -> float:
    return board_map(_spin(p, time))


@wp.func
def _cnormal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0013
    dx = _cmap(p + wp.vec3(e, 0.0, 0.0), time) - _cmap(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _cmap(p + wp.vec3(0.0, e, 0.0), time) - _cmap(p - wp.vec3(0.0, e, 0.0), time)
    dz = _cmap(p + wp.vec3(0.0, 0.0, e), time) - _cmap(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3),
                   board: wp.array(dtype=wp.vec3), helix: wp.array(dtype=wp.vec3),
                   chromo: wp.array(dtype=wp.vec3), tok: wp.array(dtype=wp.int32),
                   web: wp.array2d(dtype=wp.int32), dna: wp.array2d(dtype=wp.int32),
                   nnode: int, nweb: int, ndna: int, nbb: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, stage: int, blend: float,
                   card_fade: float, node_a: float, web_w: float, dna_w: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    col = ec.studio_sky(rd)

    # the real card (fades out as it breaks into the web)
    if card_fade > 0.01:
        t = float(0.0)
        hit = int(0)
        for _ in range(200):
            p = eye + rd * t
            d = _cmap(p, time)
            if d < 0.0007 * t + 0.0005:
                hit = 1
                break
            t += d * 0.8
            if t > _MAXD:
                break
        if hit == 1:
            p = eye + rd * t
            n = _cnormal(p, time)
            sp = _spin(p, time)
            bc = board_shade(sp, n, rd, 0.9, time)
            col = col * (1.0 - card_fade) + bc * card_fade

    # nodes: the token-words (glow), morphing board -> helix -> chromosome
    glow = wp.vec3(0.0, 0.0, 0.0)
    for k in range(nnode):
        p = _npos(k, stage, blend, board, helix, chromo)
        ge = emitter(eye, rd, p, 0.075)
        glow = glow + _tokcolor(tok[k]) * (ge * node_a * 1.5)

    # web edges (the web of words) + DNA edges (backbone + base-pair rungs)
    if web_w > 0.01:
        for e in range(nweb):
            a = _npos(web[e, 0], stage, blend, board, helix, chromo)
            b = _npos(web[e, 1], stage, blend, board, helix, chromo)
            g = _seg_glow(eye, rd, a, b, 0.05)
            glow = glow + wp.vec3(0.6, 0.85, 1.0) * (g * web_w * 0.7)
    if dna_w > 0.01:
        for e in range(ndna):
            a = _npos(dna[e, 0], stage, blend, board, helix, chromo)
            b = _npos(dna[e, 1], stage, blend, board, helix, chromo)
            if e < nbb:                                              # backbone strand — bright, thin
                g = _seg_glow(eye, rd, a, b, 0.032)
                glow = glow + wp.vec3(0.55, 0.9, 1.0) * (g * dna_w * 1.5)
            else:                                                    # base-pair rung — token-coloured
                g = _seg_glow(eye, rd, a, b, 0.05)
                tc = (_tokcolor(tok[dna[e, 0]]) + _tokcolor(tok[dna[e, 1]])) * 0.5
                glow = glow + tc * (g * dna_w * 0.9)

    img[i, j] = col + glow


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _stage(time):
    """(stage, blend, card_fade, node_alpha, web_w, dna_w) over card -> web -> DNA -> chromosome -> unwind."""
    u = (float(time) % _CYCLE) / _CYCLE
    # forward over first 0.85, then quick unwind
    if u < 0.14:                                    # the card
        return 0, 0.0, 1.0, _smooth(u / 0.14) * 0.2, 0.0, 0.0
    if u < 0.36:                                    # break into the web of words
        f = (u - 0.14) / 0.22
        return 0, 0.0, 1.0 - 0.85 * _smooth(f), _smooth(f), _smooth(f), 0.0
    if u < 0.60:                                    # read out into the DNA helix
        f = (u - 0.36) / 0.24
        return 1, _smooth(f), 0.15 * (1.0 - _smooth(f)), 1.0, 1.0 - _smooth(f), _smooth(f)
    if u < 0.85:                                    # coil into the chromosome
        f = (u - 0.60) / 0.25
        return 2, _smooth(f), 0.0, 1.0, 0.0, 1.0
    # unwind back to the card
    f = (u - 0.85) / 0.15
    return 2, 1.0 - _smooth(f), _smooth(f) * 0.9, 1.0 - 0.6 * _smooth(f), 0.0, 1.0 - _smooth(f)


def _render(width, height, time, mouse, device):
    stage, blend, card_fade, node_a, web_w, dna_w = _stage(time)
    board = wp.array(_BOARD, dtype=wp.vec3, device=device)
    helix = wp.array(_HELIX, dtype=wp.vec3, device=device)
    chromo = wp.array(_CHROMO, dtype=wp.vec3, device=device)
    tok = wp.array(_TOK, dtype=wp.int32, device=device)
    web = wp.array2d(_WEB, dtype=wp.int32, device=device)
    dna = wp.array2d(_DNA, dtype=wp.int32, device=device)

    # swing the camera to face the chromosome's X-plane as it coils (board view -> front view)
    face = blend if stage == 2 else 0.0
    az = (0.5 + 0.12 * math.sin(time * 0.2)) * (1.0 - face) + 0.03 * face + float(mouse[0]) * 0.006
    el = 0.5 * (1.0 - face) + 0.2 * face
    dist = 9.4 * (1.0 - face) + 6.6 * face
    tgt = wp.vec3(0.0, 0.1 * (1.0 - face) + 0.0 * face, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, board, helix, chromo, tok, web, dna, _N, _NWEB, _NDNA, _NBB,
                      eye, fwd, right, up, width, height, float(time), tanfov,
                      int(stage), float(blend), float(card_fade), float(node_a),
                      float(web_w), float(dna_w)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.15, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3 as a process: the real RTX board breaks into a web of token-words (each element a "
                "glowing node = its warp_compress token/value), the web is read out as a DNA double helix, "
                "and the helix coils into the four arms of a metaphase chromosome — the coiled genome is "
                "the compressed card (tokenchromo codec, lossless).",
    renderer=_render,
)
