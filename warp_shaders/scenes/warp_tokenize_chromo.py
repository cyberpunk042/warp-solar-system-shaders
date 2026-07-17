"""warp_tokenize_chromo — C3: the card's tokens become a beautiful DNA double helix that weaves itself
into a single smooth solid chromosome.

Operator directive (verbatim): *"the final product is just supposed to be a chromosome ... your DNA and
double helix too are trash, they are supposed to be beautiful and weave themselves into a chromosome ...
Even a millimeter different from this is wrong"* (with the classic blue metaphase-chromosome reference).

So the end state is ONE clean, smooth, solid chromosome — the blue X: two curved chromatid arms, fat and
rounded at the four telomere tips, pinched at the centromere with its two little nodes. The animation:

  1. **The card** (real `gpu_board`).
  2. **~A million tokens** — the board is voxelised fine (~1.2M-cell grid), painted as that grid.
  3. **DNA double helix** — the tokens rise into a beautiful spinning double helix (two backbones +
     coloured base-pair rungs), grown from the card's own tokens.
  4. **Weave** — the helix coils and condenses, weaving itself into the chromosome.
  5. **Chromosome** — a single smooth solid blue X, slowly turning. Then it unwinds back to the card.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.sdf import sd_capsule, sd_sphere, op_smooth_union
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card

_MAXD = 40.0
_CYCLE = 20.0
_BLOCK = 2
_UP = 3                           # voxel upsample per axis -> ~1.2M token cells (a real million)
_X0, _Y0, _Z0 = -3.7, -0.14, -1.5
_CARDY = 0.14
# the beautiful DNA double helix (vertical, centred)
_NH = 150                         # helix nodes per backbone
_HH = 2.9                         # helix height (== chromosome height)
_RHX = 0.52                       # helix radius
_THX = 4.5                        # helix turns (open, elegant — a twisted ladder)
_TR = 0.060                       # backbone tube radius
_RR = 0.040                       # base-pair rung radius
_RUNGSTEP = 2
# the solid chromosome (the blue X — the end state)
_CTX, _CTY = 0.52, 1.45           # half-width / half-height (telomere reach); height 2*_CTY == _HH
_CK = 0.34                        # smooth-union blend (the smooth solid body + pinched centromere)
_CBODY = wp.vec3(0.15, 0.46, 0.72)      # chromosome blue
_CNODE = wp.vec3(0.62, 0.82, 0.94)      # lighter centromere nodes
_BASECOL = ((0.92, 0.22, 0.20), (0.24, 0.86, 0.34), (0.26, 0.42, 0.96), (0.96, 0.86, 0.22))


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

    # ~a million tokens: fine token grid over the card, coloured by each element's token
    nx0, ny0, nz0 = occ.shape
    b5 = 5
    _, idx5, _ = mc.compress(occ, block=b5)
    ii = np.minimum(np.arange(nx0)[:, None, None] // b5, idx5.shape[0] - 1)
    jj = np.minimum(np.arange(ny0)[None, :, None] // b5, idx5.shape[1] - 1)
    kk = np.minimum(np.arange(nz0)[None, None, :] // b5, idx5.shape[2] - 1)
    tokvox = np.where(occ > 0, idx5[ii, jj, kk], -1).astype(np.int32)
    tok3d = np.repeat(np.repeat(np.repeat(tokvox, _UP, 0), _UP, 1), _UP, 2).astype(np.int32)

    # the helix backbone colours come from the card's tokens, in proximity order (grown from the card)
    bwx, bwz = 7.4 / nbx, 3.0 / nbz
    order = []
    for bk in range(nbz):
        rng = range(nbx) if (bk % 2 == 0) else range(nbx - 1, -1, -1)
        for bi in rng:
            if occ_col[bi, bk]:
                order.append((bi, bk))
    npath = len(order)
    idxs = np.linspace(0, npath - 1, _NH).round().astype(int)
    cardxyz = np.zeros((_NH, 3), np.float32)
    col = np.zeros((_NH, 3), np.float32)
    for k, j in enumerate(idxs):
        bi, bk = order[j]
        cardxyz[k] = (-3.7 + (bi + 0.5) * bwx, _CARDY, -1.5 + (bk + 0.5) * bwz)
        col[k] = _hue_np(int(tok_col[bi, bk]))
    return (np.ascontiguousarray(cardxyz), np.ascontiguousarray(col), npath,
            np.ascontiguousarray(tok3d))


def _helix():
    """a clean, beautiful vertical double helix — two backbones 180 deg apart, base-pair rungs across."""
    t = np.linspace(0.0, 1.0, _NH)
    y = (t - 0.5) * _HH
    ang = 2.0 * math.pi * _THX * t
    ca, sa = np.cos(ang), np.sin(ang)
    pa = np.stack([_RHX * ca, y, _RHX * sa], 1).astype(np.float32)
    pb = np.stack([-_RHX * ca, y, -_RHX * sa], 1).astype(np.float32)
    rung_idx = np.array([i for i in range(_NH) if i % _RUNGSTEP == 0], np.int32)
    rcol = np.array([_BASECOL[k % 4] for k in range(len(rung_idx))], np.float32)
    return np.ascontiguousarray(pa), np.ascontiguousarray(pb), rung_idx, np.ascontiguousarray(rcol)


_CARDXYZ, _COL, _NPATH, _TOK3D = _build()
_NX, _NY, _NZ = _TOK3D.shape
_CX, _CY, _CZ = 7.4 / _NX, 0.44 / _NY, 3.0 / _NZ
_NTOK = int((_TOK3D >= 0).sum())
_PA, _PB, _RUNG_IDX, _RCOL = _helix()
_NR = int(_RUNG_IDX.shape[0])
# the card positions for the rung nodes, for the lift/assembly
_CARD_RUNG = np.ascontiguousarray(_CARDXYZ[_RUNG_IDX])


def _ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _positions(lift):
    """the helix lifts up out of the card (lift 0 -> the tokens still on the board; 1 -> full helix)."""
    a = _ss(lift)
    pa = (_CARDXYZ * (1.0 - a) + _PA * a).astype(np.float32)
    pb = (_CARDXYZ * (1.0 - a) + _PB * a).astype(np.float32)
    rad = np.full(_NH, _TR * a + 0.004, np.float32)
    ra = pa[_RUNG_IDX]
    rb = pb[_RUNG_IDX]
    rr = np.full(_NR, _RR * a, np.float32)
    return pa, pb, rad, ra, rb, rr


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
def _voxtok(p: wp.vec3, tok: wp.array3d(dtype=wp.int32), nx: int, ny: int, nz: int) -> int:
    i = int(wp.clamp((p[0] - _X0) / _CX, 0.0, float(nx - 1)))
    j = int(wp.clamp((p[1] - _Y0) / _CY, 0.0, float(ny - 1)))
    k = int(wp.clamp((p[2] - _Z0) / _CZ, 0.0, float(nz - 1)))
    return tok[i, j, k]


@wp.func
def _seam(p: wp.vec3) -> float:
    fx = (p[0] - _X0) / _CX
    fz = (p[2] - _Z0) / _CZ
    ex = wp.abs(fx - wp.floor(fx) - 0.5)
    ez = wp.abs(fz - wp.floor(fz) - 0.5)
    e = wp.max(ex, ez)
    return wp.clamp((0.5 - e) * 6.0, 0.55, 1.0)


@wp.func
def _arm(p: wp.vec3, tip: wp.vec3, k: float) -> float:
    """one fat, rounded chromatid arm: thin at the centromere, bulging to a fat rounded telomere tip."""
    c = wp.vec3(0.0, 0.0, 0.0)
    d = sd_capsule(p, c, tip, 0.095)                     # thin core -> keeps the centromere pinched
    d = op_smooth_union(d, sd_sphere(p - tip * 0.35, 0.19), k)   # inner-arm swell (slender)
    d = op_smooth_union(d, sd_sphere(p - tip * 0.65, 0.29), k)   # mid-arm bulge
    d = op_smooth_union(d, sd_sphere(p - tip, 0.36), k)          # fat rounded telomere tip
    return d


@wp.func
def _chromo(p: wp.vec3) -> float:
    """the solid chromosome — a smooth blue X: four fat arms pinched at the centromere."""
    k = _CK
    d = _arm(p, wp.vec3(-_CTX, _CTY, 0.0), k)
    d = op_smooth_union(d, _arm(p, wp.vec3(_CTX, _CTY, 0.0), k), k)
    d = op_smooth_union(d, _arm(p, wp.vec3(-_CTX, -_CTY, 0.0), k), k)
    d = op_smooth_union(d, _arm(p, wp.vec3(_CTX, -_CTY, 0.0), k), k)
    return d


@wp.func
def _nodes(p: wp.vec3) -> float:
    """the two little centromere nodes at the waist."""
    d = sd_sphere(p - wp.vec3(-0.19, 0.0, 0.20), 0.155)
    d = wp.min(d, sd_sphere(p - wp.vec3(0.19, 0.0, 0.20), 0.155))
    return d


@wp.func
def _tube(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
          rad: wp.array(dtype=float), n: int) -> float:
    best = _MAXD
    for i in range(n - 1):
        r = rad[i]
        if r > 0.006:
            d = sd_capsule(p, pa[i], pa[i + 1], r)
            if d < best:
                best = d
            d = sd_capsule(p, pb[i], pb[i + 1], r)
            if d < best:
                best = d
    return best


@wp.func
def _tubeseg(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
             rad: wp.array(dtype=float), n: int) -> int:
    best = _MAXD
    seg = int(0)
    for i in range(n - 1):
        r = rad[i]
        if r > 0.006:
            d = sd_capsule(p, pa[i], pa[i + 1], r)
            if d < best:
                best = d
                seg = i
            d = sd_capsule(p, pb[i], pb[i + 1], r)
            if d < best:
                best = d
                seg = i
    return seg


@wp.func
def _rungs(p: wp.vec3, ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3),
           rr: wp.array(dtype=float), nr: int) -> float:
    best = _MAXD
    for k in range(nr):
        if rr[k] > 0.006:
            d = sd_capsule(p, ra[k], rb[k], rr[k])
            if d < best:
                best = d
    return best


@wp.func
def _rungseg(p: wp.vec3, ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3),
             rr: wp.array(dtype=float), nr: int) -> int:
    best = _MAXD
    seg = int(0)
    for k in range(nr):
        if rr[k] > 0.006:
            d = sd_capsule(p, ra[k], rb[k], rr[k])
            if d < best:
                best = d
                seg = k
    return seg


@wp.func
def _map(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
         rad: wp.array(dtype=float), n: int,
         ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float), nr: int,
         cerode: float, shide: float, weave: float) -> float:
    d = _MAXD
    if cerode < 1.6:
        d = wp.min(d, board_map(p) + cerode)
    if shide < 1.6:
        dh = _tube(p, pa, pb, rad, n)                         # the DNA double helix
        dc = op_smooth_union(_chromo(p), _nodes(p), 0.10)     # the solid chromosome
        d = wp.min(d, (dh * (1.0 - weave) + dc * weave) + shide)   # weave the helix into the chromosome
        d = wp.min(d, _rungs(p, ra, rb, rr, nr) + shide + weave * 8.0)   # base pairs fade as it weaves
    return d


@wp.func
def _normal(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
            rad: wp.array(dtype=float), n: int,
            ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float), nr: int,
            cerode: float, shide: float, weave: float) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave) - _map(p - wp.vec3(e, 0.0, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
    dy = _map(p + wp.vec3(0.0, e, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave) - _map(p - wp.vec3(0.0, e, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
    dz = _map(p + wp.vec3(0.0, 0.0, e), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave) - _map(p - wp.vec3(0.0, 0.0, e), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, nrm: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
        rad: wp.array(dtype=float), n: int,
        ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float), nr: int,
        cerode: float, shide: float, weave: float) -> float:
    o = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.05 * float(k)
        d = _map(p + nrm * hr, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
        o += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * o, 0.0, 1.0)


@wp.func
def _shade(tc: wp.vec3, nrm: wp.vec3, rd: wp.vec3, ao: float) -> wp.vec3:
    """organic two-light shading: warm key + cool fill + fresnel rim + glossy highlight."""
    key = wp.normalize(wp.vec3(0.5, 0.82, 0.42))
    fill = wp.normalize(wp.vec3(-0.5, 0.28, -0.55))
    klit = wp.clamp(wp.dot(nrm, key), 0.0, 1.0)
    flit = wp.clamp(wp.dot(nrm, fill), 0.0, 1.0)
    ao2 = ao * ao
    warm = wp.vec3(tc[0] * 1.00, tc[1] * 0.95, tc[2] * 0.85)
    cool = wp.vec3(tc[0] * 0.55, tc[1] * 0.70, tc[2] * 1.00)
    body = warm * ((0.14 + 0.90 * klit) * ao2) + cool * (0.32 * flit * ao2)
    hv = wp.normalize(key - rd)
    spec = wp.pow(wp.clamp(wp.dot(nrm, hv), 0.0, 1.0), 40.0) * 1.0
    rim = wp.pow(1.0 - wp.clamp(-wp.dot(nrm, rd), 0.0, 1.0), 2.5) * 0.40
    return body + wp.vec3(spec, spec, spec) + tc * rim


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), tok: wp.array3d(dtype=wp.int32),
                   nx: int, ny: int, nz: int,
                   pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
                   rad: wp.array(dtype=float), col: wp.array(dtype=wp.vec3), n: int,
                   ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float),
                   rcol: wp.array(dtype=wp.vec3), nr: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, cerode: float, shide: float, tokamt: float, weave: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    tsky = wp.clamp(rd[1] * 0.6 + 0.42, 0.0, 1.0)
    bg = wp.vec3(0.018, 0.024, 0.040) * (1.0 - tsky) + wp.vec3(0.050, 0.060, 0.095) * tsky

    t = float(0.0)
    hit = int(0)
    for _ in range(220):
        p = eye + rd * t
        d = _map(p, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
        if d < 0.0006 * t + 0.0004:
            hit = 1
            break
        t += d * 0.80
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = bg
        return

    p = eye + rd * t
    nrm = _normal(p, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
    ao = _ao(p, nrm, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide, weave)
    lit = wp.clamp(wp.dot(nrm, wp.normalize(wp.vec3(0.45, 0.8, 0.45))), 0.2, 1.0)
    fog = wp.exp(-0.022 * t)
    dcard = _MAXD
    if cerode < 1.6:
        dcard = board_map(p) + cerode
    dstr = _MAXD
    drung = _MAXD
    if shide < 1.6:
        dh = _tube(p, pa, pb, rad, n)
        dc = op_smooth_union(_chromo(p), _nodes(p), 0.10)
        dstr = (dh * (1.0 - weave) + dc * weave) + shide
        drung = _rungs(p, ra, rb, rr, nr) + shide + weave * 8.0

    if drung <= dcard and drung <= dstr:
        bc = rcol[_rungseg(p, ra, rb, rr, nr)]                # a base-pair rung (A/T/G/C)
        img[i, j] = _shade(bc, nrm, rd, ao) * fog + bg * (1.0 - fog)
        return
    if dstr <= dcard:
        # DNA token colour weaves to chromosome blue; the two centromere nodes are lighter blue
        tokc = col[_tubeseg(p, pa, pb, rad, n)]
        bluec = _CBODY
        if weave > 0.4:
            if _nodes(p) < _chromo(p) - 0.02:
                bluec = _CNODE
        base = tokc * (1.0 - weave) + bluec * weave
        img[i, j] = _shade(base, nrm, rd, ao) * fog + bg * (1.0 - fog)
        return

    # the card surface — painted as ~a million token cells
    bshade = board_shade(p, nrm, rd, ao, time)
    tc = _tokcolor(_voxtok(p, tok, nx, ny, nz))
    toklook = tc * ((0.5 + 0.55 * lit) * ao) * _seam(p)
    img[i, j] = (bshade * (1.0 - tokamt) + toklook * tokamt) * fog + bg * (1.0 - fog)


def _fwd(s):
    """forward half s in [0,1] -> (lift, cerode, shide, tokamt, weave). card -> million tokens ->
    the tokens rise into a beautiful DNA double helix -> it weaves into the solid chromosome."""
    if s < 0.08:                                    # 1. the real card
        return 0.0, 0.0, 3.0, 0.0, 0.0
    if s < 0.20:                                    # 2. the card becomes ~a million tokens
        return 0.0, 0.0, 3.0, _ss((s - 0.08) / 0.12), 0.0
    if s < 0.28:                                    #    hold the million-token card
        return 0.0, 0.0, 3.0, 1.0, 0.0
    if s < 0.50:                                    # 3. the tokens rise into the DNA double helix
        f = _ss((s - 0.28) / 0.22)
        return f, 2.0 * f, 3.0 * (1.0 - f), 1.0, 0.0
    if s < 0.60:                                    #    admire the helix
        return 1.0, 2.0, 0.0, 1.0, 0.0
    if s < 0.85:                                    # 4. the helix weaves itself into the chromosome
        return 1.0, 2.0, 0.0, 1.0, _ss((s - 0.60) / 0.25)
    return 1.0, 2.0, 0.0, 1.0, 1.0                  # 5. the solid chromosome (held, turning)


def _state(time):
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    lift, cerode, shide, tokamt, weave = _state(time)
    posA, posB, radn, raa, rbb, rrr = _positions(lift)
    pa = wp.array(posA, dtype=wp.vec3, device=device)
    pb = wp.array(posB, dtype=wp.vec3, device=device)
    rad = wp.array(radn, dtype=float, device=device)
    col = wp.array(_COL, dtype=wp.vec3, device=device)
    ra = wp.array(raa, dtype=wp.vec3, device=device)
    rb = wp.array(rbb, dtype=wp.vec3, device=device)
    rr = wp.array(rrr, dtype=float, device=device)
    rcol = wp.array(_RCOL, dtype=wp.vec3, device=device)
    tok = wp.array3d(_TOK3D, dtype=wp.int32, device=device)

    # continuous slow orbit so the helix + chromosome clearly turn (a real animation, not static)
    az = 0.5 + time * 0.30 + float(mouse[0]) * 0.006
    el = 0.42 * (1.0 - lift) + 0.06 * lift
    dist = 9.0 * (1.0 - lift) + 6.6 * lift
    tgt = wp.vec3(-0.1 * (1.0 - lift), 0.15 * (1.0 - lift), 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, tok, _NX, _NY, _NZ, pa, pb, rad, col, _NH, ra, rb, rr, rcol, _NR,
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(cerode), float(shide), float(tokamt), float(weave)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.6, strength=0.6, radius=6, passes=4)
    out = post.tonemap(hdr, mode="aces", exposure=1.18, preserve_hue=True)
    out = post.vignette(out, amount=0.32)
    return out


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3: the real RTX board becomes ~a million token cells that rise into a beautiful DNA "
                "double helix (two backbones + coloured base-pair rungs, grown from the card's tokens), "
                "which weaves itself into a single smooth solid chromosome — the blue metaphase X — then "
                "unwinds back to the card.",
    renderer=_render,
)
