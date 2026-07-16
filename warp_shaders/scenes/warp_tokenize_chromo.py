"""warp_tokenize_chromo — C3, from scratch: the card's tokens ARE the genome; it condenses into a chromosome.

Operator arc (verbatim): turn the card into tokens; the tokens *"connect with the proximity token(s) to
form a DNA strain, continuous strain ... base pair merging then double helix and tighter and tighter till
you can center it at the telomere into the chromosome"*; and *"there is so much more token in a card... do
it right."*

Built ONE way — bottom-up. There is a single continuous strand whose material is the card's own tokens,
threaded in **proximity order** (a serpentine adjacency walk through the real token cells, so neighbours on
the strand are neighbours on the card). That one strand is never a fabricated shape poured full — it is the
tokens — and it moves through the real chromosome hierarchy:

  1. **In place on the card** — the strand lies on the board as the serpentine through its tokens.
  2. **Gather → DNA double helix** — the board erodes and the strand lifts and winds into a double helix,
     two backbones of the card's tokens, base pairs between them.
  3. **Condense → chromosome X** — the helix folds into two chromatids crossing at the centromere and coils
     **tighter and tighter** (super density) into the metaphase **X**.
  4. **Reverse** — it unwinds all the way back to the card. `time` runs the whole cycle, then loops.

Every colour along the strand is a real card token, in the card's own adjacency order — the compression is
OF the card, grown from it, not a picture of DNA with the card poured in.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..procedural.sdf import sd_capsule
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card

_MAXD = 40.0
_CYCLE = 16.0
_BLOCK = 2
_N = 108                          # strand samples (one continuous genome, subsampled from the card tokens)
_TR = 0.052                       # strand tube radius
_YC = 0.95                        # the strand/chromosome floats above the card footprint as it lifts out
_CARDY = 0.06                     # the strand's in-place height on the card
_LX = 3.0                         # DNA extent along x
_RD = 0.42                        # DNA helix radius
_DNAT = 15.0                      # DNA turns over the length
_AX = 0.72                        # chromosome X half-width
_AY = 1.18                        # chromosome X half-height
_RC = 0.15                        # condensed-coil radius
_XT = 40.0                        # condensed-coil turns (super density)


def _hue_np(tid):
    if tid < 0:
        return (0.10, 0.13, 0.12)
    h = (float(tid) * 0.61803) % 1.0
    r = min(max(abs(h * 6.0 - 3.0) - 1.0, 0.0), 1.0)
    g = min(max(2.0 - abs(h * 6.0 - 2.0), 0.0), 1.0)
    b = min(max(2.0 - abs(h * 6.0 - 4.0), 0.0), 1.0)
    return (r, g, b)


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
    occ_col = occ_blocks.any(axis=1)
    tok_col = np.full((nbx, nbz), -1, np.int32)
    for by in range(nby - 1, -1, -1):
        has = occ_blocks[:, by, :]
        f = (tok_col < 0) & has
        tok_col[f] = index[:, by, :][f]
    bwx = 7.4 / nbx
    bwz = 3.0 / nbz
    # serpentine adjacency path through the occupied token cells -> proximity order of the genome
    order = []
    for bk in range(nbz):
        rng = range(nbx) if (bk % 2 == 0) else range(nbx - 1, -1, -1)
        for bi in rng:
            if occ_col[bi, bk]:
                order.append((bi, bk))
    npath = len(order)
    # subsample the genome to _N nodes: in-place card positions + token colour per node
    idx = np.linspace(0, npath - 1, _N).round().astype(int)
    cardxyz = np.zeros((_N, 3), np.float32)
    col = np.zeros((_N, 3), np.float32)
    for k, j in enumerate(idx):
        bi, bk = order[j]
        cardxyz[k] = (-3.7 + (bi + 0.5) * bwx, _CARDY, -1.5 + (bk + 0.5) * bwz)
        col[k] = _hue_np(int(tok_col[bi, bk]))
    return np.ascontiguousarray(cardxyz), np.ascontiguousarray(col), npath


_CARDXYZ, _COL, _NPATH = _build()


def _ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _xbase(t):
    """the metaphase X traced by one continuous strand: two chromatids (diagonals) crossing at centre."""
    out = np.zeros((t.shape[0], 3), np.float64)
    for k in range(t.shape[0]):
        tt = t[k]
        if tt < 0.5:
            p0 = np.array([-_AX, _AY, 0.0]); p2 = np.array([_AX, -_AY, 0.0]); lt = tt * 2.0
        else:
            p0 = np.array([_AX, _AY, 0.0]); p2 = np.array([-_AX, -_AY, 0.0]); lt = (tt - 0.5) * 2.0
        c = np.array([0.0, 0.0, 0.0])
        if lt < 0.5:
            out[k] = p0 * (1.0 - lt * 2.0) + c * (lt * 2.0)
        else:
            out[k] = c * (1.0 - (lt - 0.5) * 2.0) + p2 * ((lt - 0.5) * 2.0)
    out[:, 1] += _YC
    return out


def _positions(a):
    """the one strand at morph a in [0,2]: 0 = card serpentine, 1 = DNA helix, 2 = chromosome X."""
    t = np.linspace(0.0, 1.0, _N)
    dna_base = np.stack([(t - 0.5) * 2.0 * _LX, np.full(_N, _YC), np.zeros(_N)], 1)
    xb = _xbase(t)
    if a <= 1.0:
        f = _ss(a)
        base = _CARDXYZ.astype(np.float64) * (1.0 - f) + dna_base * f
        wamp = f * _RD
        turns = _DNAT
    else:
        f = _ss(a - 1.0)
        base = dna_base * (1.0 - f) + xb * f
        wamp = _RD * (1.0 - f) + _RC * f
        turns = _DNAT * (1.0 - f) + _XT * f
    phi = 2.0 * math.pi * turns * t
    wa = np.stack([np.zeros(_N), np.cos(phi), np.sin(phi)], 1) * wamp
    wb = np.stack([np.zeros(_N), np.cos(phi + math.pi), np.sin(phi + math.pi)], 1) * wamp
    return (base + wa).astype(np.float32), (base + wb).astype(np.float32)


# --------------------------------------------------------------------------- strand SDF (the genome tube)
@wp.func
def _tube(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3), n: int) -> float:
    best = _MAXD
    for i in range(n - 1):
        d = sd_capsule(p, pa[i], pa[i + 1], _TR)
        if d < best:
            best = d
        d = sd_capsule(p, pb[i], pb[i + 1], _TR)
        if d < best:
            best = d
    return best


@wp.func
def _tubeseg(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3), n: int) -> int:
    best = _MAXD
    seg = int(0)
    for i in range(n - 1):
        d = sd_capsule(p, pa[i], pa[i + 1], _TR)
        if d < best:
            best = d
            seg = i
        d = sd_capsule(p, pb[i], pb[i + 1], _TR)
        if d < best:
            best = d
            seg = i
    return seg


@wp.func
def _map(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3), n: int, cerode: float) -> float:
    d = _tube(p, pa, pb, n)
    if cerode < 1.6:
        d = wp.min(d, board_map(p) + cerode)
    return d


@wp.func
def _normal(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3), n: int, cerode: float) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0), pa, pb, n, cerode) - _map(p - wp.vec3(e, 0.0, 0.0), pa, pb, n, cerode)
    dy = _map(p + wp.vec3(0.0, e, 0.0), pa, pb, n, cerode) - _map(p - wp.vec3(0.0, e, 0.0), pa, pb, n, cerode)
    dz = _map(p + wp.vec3(0.0, 0.0, e), pa, pb, n, cerode) - _map(p - wp.vec3(0.0, 0.0, e), pa, pb, n, cerode)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, nrm: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
        n: int, cerode: float) -> float:
    o = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.05 * float(k)
        d = _map(p + nrm * hr, pa, pb, n, cerode)
        o += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * o, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
                   col: wp.array(dtype=wp.vec3), n: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, cerode: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(200):
        p = eye + rd * t
        d = _map(p, pa, pb, n, cerode)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.85
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    nrm = _normal(p, pa, pb, n, cerode)
    ao = _ao(p, nrm, pa, pb, n, cerode)
    lit = wp.clamp(wp.dot(nrm, wp.normalize(wp.vec3(0.4, 0.85, 0.5))), 0.2, 1.0)
    dtube = _tube(p, pa, pb, n)
    dcard = _MAXD
    if cerode < 1.6:
        dcard = board_map(p) + cerode

    if dtube <= dcard:
        seg = _tubeseg(p, pa, pb, n)
        tc = col[seg]                                        # the card token this stretch of strand is
        img[i, j] = tc * ((0.5 + 0.6 * lit) * ao) + tc * 0.12
        return

    img[i, j] = board_shade(p, nrm, rd, ao, time)            # the surviving card, eroding away


def _fwd(s):
    """forward half s in [0,1] -> (a, cerode). card -> gather to DNA -> fold+coil to X -> hold."""
    if s < 0.14:
        return 0.0, 0.0
    if s < 0.46:
        f = _ss((s - 0.14) / 0.32)
        return f, 2.0 * f
    if s < 0.82:
        f = _ss((s - 0.46) / 0.36)
        return 1.0 + f, 2.0
    return 2.0, 2.0


def _state(time):
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    a, cerode = _state(time)
    posA, posB = _positions(a)
    pa = wp.array(posA, dtype=wp.vec3, device=device)
    pb = wp.array(posB, dtype=wp.vec3, device=device)
    col = wp.array(_COL, dtype=wp.vec3, device=device)

    lift = min(1.0, a)                                        # 0 on the card, 1 once lifted into the strand
    az = 0.55 + 0.06 * math.sin(time * 0.12) + float(mouse[0]) * 0.006
    el = 0.42 * (1.0 - lift) + 0.20 * lift
    dist = 9.0 * (1.0 - lift) + 6.4 * lift
    tgt = wp.vec3(-0.1 * (1.0 - lift), 0.15 * (1.0 - lift) + _YC * lift, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, pa, pb, col, _N, eye, fwd, right, up, width, height,
                      float(time), tanfov, float(cerode)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3, from scratch: a single continuous strand whose material is the card's own tokens, "
                "threaded in proximity order, lies on the board, then lifts and winds into a DNA double "
                "helix and folds + coils tighter and tighter into the metaphase chromosome X — the card's "
                "tokens condensed into a chromosome, grown from the card, not a picture of DNA poured full; "
                "then it unwinds back to the card.",
    renderer=_render,
)
