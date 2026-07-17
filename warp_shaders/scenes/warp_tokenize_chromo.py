"""warp_tokenize_chromo — C3: the card -> ~a MILLION tokens -> the chromosome-packing JOURNEY (one chain).

Operator directives (verbatim): *"TURN THE CARD INTO 1 MILLION TOKEN.. LET THE TOKEN ONE BY ONE
CONNECT WITH EACH OFTHER TO FORM DNA BRANCH. ONE BY ONE let the branches connect to each other to form
a double-helix ... the double helix have to become nucleosomes to form telomere and to form the
chromosome ... nothing can be skipped or be done in parallel ... the double HELIXes ... everything is
plural ... it is apart before being one ... did you visualize the chain? the journey?"*

C3 is the packing-diagram JOURNEY as one continuous chain grown from the card's tokens. Read it from
the loose end (bottom-left) up to the packed end (top-right) and you watch DNA become a chromosome,
level by level, in strict order, nothing skipped, nothing parallel:

  1. **The card** (real `gpu_board`).
  2. **~A million tokens** — the board is voxelised fine (~1.2M-cell grid), painted as that grid.
  3. **Base pairs -> double helix** — tokens gather (proximity order) into two backbones; **base-pair
     rungs** (A/T/G/C colours) bridge them, one token connecting to the next, winding into the helix.
  4. **Nucleosomes** — the helix beads up onto histone-like cores (beads on a string).
  5. **Chromosome** — the beaded string coils into the **two chromatids** of the metaphase **X** (plural
     coiled arms — apart — meeting at the one centromere; telomere tips).

The chain grows out of the card loose->packed over `time`, each level forming in strict order; the held
state shows the whole journey at once. Then it unwinds back to the card.
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
_CYCLE = 20.0
_BLOCK = 2
_UP = 3                           # voxel upsample per axis -> ~1.2M token cells (a real million)
_X0, _Y0, _Z0 = -3.7, -0.14, -1.5
_N = 156                          # strand nodes along the whole journey
_TR = 0.052                       # base tube radius (one double-helix backbone)
_YC = 0.95                        # journey height
_CARDY = 0.14
_MAXSEG = 0.7                     # skip capsules longer than this (only the centromere crossover)
_NPER = 4.0                       # strand nodes per nucleosome bead
_NBUMP = 1.4                      # how much the tube bulges into a bead
_ZH = 0.30                        # zone: double helix  (0 .. _ZH)
_ZNU = 0.52                       # zone: nucleosomes   (_ZH .. _ZNU) ; chromosome X (_ZNU .. 1)
_STARTX = -5.4
_LXH = 2.8                        # x-length of the loose double-helix run
_LXN = 2.2                        # x-length of the nucleosome run
_HRISE = 0.6                      # y the helix climbs over its run
_NRISE = 0.95                     # y the nucleosome string climbs into the chromosome
_AXW, _AYW = 0.94, 1.38           # chromosome-X arm half-extents (telomere reach)
_RCH = 0.17                       # chromatid solenoid radius (the coiled arm)
_KCH = 6.0                        # chromatid solenoid turns per arm (dense)
_RUNGT = 0.30                     # base-pair rungs are drawn across the loose double helix (t < this)
_RUNGSTEP = 2                     # a base-pair rung every this-many nodes
_RR = 0.030                       # base-pair rung radius (thin)
# A / T / G / C base-pair colours (the diagram's coloured rungs)
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

    bwx = 7.4 / nbx
    bwz = 3.0 / nbz
    order = []
    for bk in range(nbz):
        rng = range(nbx) if (bk % 2 == 0) else range(nbx - 1, -1, -1)
        for bi in rng:
            if occ_col[bi, bk]:
                order.append((bi, bk))
    npath = len(order)
    idxs = np.linspace(0, npath - 1, _N).round().astype(int)
    cardxyz = np.zeros((_N, 3), np.float32)
    col = np.zeros((_N, 3), np.float32)
    for k, j in enumerate(idxs):
        bi, bk = order[j]
        cardxyz[k] = (-3.7 + (bi + 0.5) * bwx, _CARDY, -1.5 + (bk + 0.5) * bwz)
        col[k] = _hue_np(int(tok_col[bi, bk]))
    return (np.ascontiguousarray(cardxyz), np.ascontiguousarray(col), npath,
            np.ascontiguousarray(tok3d))


def _bead(i, amt=1.0):
    return _TR * (1.0 + amt * _NBUMP * (0.5 + 0.5 * math.cos(2.0 * math.pi * i / _NPER)))


def _coiled_arm(p0, p1, n):
    """a dense solenoid (coiled chromatin) from p0 to p1 — one chromatid arm."""
    u = np.linspace(0.0, 1.0, n)
    axis = p0[None, :] * (1.0 - u)[:, None] + p1[None, :] * u[:, None]
    d = p1 - p0; d = d / (np.linalg.norm(d) + 1e-9)
    e1 = np.cross(d, np.array([0.0, 0.0, 1.0])); e1 = e1 / (np.linalg.norm(e1) + 1e-9)
    e2 = np.cross(d, e1)
    ph = 2.0 * math.pi * _KCH * u
    coil = _RCH * (np.cos(ph)[:, None] * e1[None, :] + np.sin(ph)[:, None] * e2[None, :])
    return axis + coil


def _chain_struct():
    """the fixed full journey as ONE continuous thread, stitched end-to-end: double helix -> nucleosome
    beads -> the two coiled chromatids of the chromosome X. Built once."""
    N = _N
    t = np.linspace(0.0, 1.0, N)
    n_h = int(np.sum(t < _ZH))                      # double-helix nodes
    n_nu = int(np.sum((t >= _ZH) & (t < _ZNU)))     # nucleosome nodes
    n_x = N - n_h - n_nu                            # chromosome nodes
    # zone 1 — the loose double helix, climbing gently
    uh = np.linspace(0.0, 1.0, n_h)
    helix = np.stack([_STARTX + _LXH * uh, _YC + _HRISE * uh, np.zeros(n_h)], 1)
    # zone 2 — nucleosome string, continues from the helix end and climbs into the chromosome
    un = np.linspace(0.0, 1.0, n_nu)
    h0 = helix[-1]
    nucleo = np.stack([h0[0] + _LXN * un, h0[1] + _NRISE * un, np.zeros(n_nu)], 1)
    # zone 3 — chromosome X: two dense coiled chromatid arms crossing at the centromere, entry at origin
    a2, y2 = 2.0 * _AXW, 2.0 * _AYW
    n_arm = n_x // 2
    armA = _coiled_arm(np.array([0.0, 0.0, 0.0]), np.array([a2, y2, 0.0]), n_arm)        # "/" through centre
    armB = _coiled_arm(np.array([0.0, y2, 0.0]), np.array([a2, 0.0, 0.0]), n_x - n_arm)  # "\" through centre
    xloc = np.vstack([armA, armB])
    xg = xloc + (nucleo[-1] - xloc[0])              # stitch: chromosome entry == nucleosome end
    sp = np.vstack([helix, nucleo, xg])

    wamp = np.zeros(N)
    td = np.zeros(N)                               # fine double-helix turn-density
    rad = np.full(N, _TR)
    wamp[:n_h] = 0.34; td[:n_h] = 1.4
    for i in range(n_h, n_h + n_nu):                # nucleosomes: backbones merge into one beaded string
        wamp[i] = 0.05; td[i] = 1.1; rad[i] = _bead(i, 1.8)
    for i in range(n_h + n_nu, N):                  # chromosome: thick coiled rod, faint bead texture
        wamp[i] = 0.0; td[i] = 0.0
        rad[i] = _TR * 1.75 * (1.0 + 0.22 * (0.5 + 0.5 * math.cos(2.0 * math.pi * float(i) / _NPER)))

    # winding: perpendicular frame along the spine, phase from cumulative turn-density
    tan = np.zeros_like(sp)
    tan[1:-1] = sp[2:] - sp[:-2]
    tan[0] = sp[1] - sp[0]
    tan[-1] = sp[-1] - sp[-2]
    tan /= (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (N, 1))
    par = np.abs((tan * ref).sum(1)) > 0.9
    ref[par] = np.array([1.0, 0.0, 0.0])
    uu = np.cross(tan, ref); uu /= (np.linalg.norm(uu, axis=1, keepdims=True) + 1e-9)
    vv = np.cross(tan, uu)
    seg = np.zeros(N)
    seg[1:] = np.linalg.norm(sp[1:] - sp[:-1], axis=1)
    phi = np.cumsum(2.0 * math.pi * td * seg)
    off = wamp[:, None] * (np.cos(phi)[:, None] * uu + np.sin(phi)[:, None] * vv)

    # base-pair rungs across the loose double helix (a coloured A/T/G/C every few nodes)
    rung_idx = [i for i in range(N) if t[i] < _RUNGT and i % _RUNGSTEP == 0]
    rcol = np.array([_BASECOL[k % 4] for k in range(len(rung_idx))], np.float32)
    return (np.ascontiguousarray(sp), np.ascontiguousarray(off),
            np.ascontiguousarray(rad.astype(np.float32)), np.ascontiguousarray(t),
            np.array(rung_idx, np.int32), np.ascontiguousarray(rcol))


_CARDXYZ, _COL, _NPATH, _TOK3D = _build()
_NX, _NY, _NZ = _TOK3D.shape
_CX, _CY, _CZ = 7.4 / _NX, 0.44 / _NY, 3.0 / _NZ
_NTOK = int((_TOK3D >= 0).sum())
_SPINE, _OFF, _RAD, _TT, _RUNG_IDX, _RCOL = _chain_struct()
_NR = int(_RUNG_IDX.shape[0])


def _ss(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _positions(reveal):
    """grow the journey from the card: node i assembles once the reveal front passes its t."""
    asm = np.clip((reveal - _TT + 0.09) / 0.06, 0.0, 1.0)   # front over-runs so the tail fully assembles
    asm = asm * asm * (3.0 - 2.0 * asm)
    a = asm[:, None]
    base = _CARDXYZ.astype(np.float64) * (1.0 - a) + _SPINE * a
    off = _OFF * a
    pa = (base + off).astype(np.float32)
    pb = (base - off).astype(np.float32)
    rad = (_RAD * asm).astype(np.float32)
    ra = pa[_RUNG_IDX]
    rb = pb[_RUNG_IDX]
    rr = (_RR * asm[_RUNG_IDX]).astype(np.float32)
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
def _tube(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
          rad: wp.array(dtype=float), n: int) -> float:
    best = _MAXD
    for i in range(n - 1):
        r = rad[i]
        if r > 0.006:
            if wp.length(pa[i + 1] - pa[i]) < _MAXSEG:
                d = sd_capsule(p, pa[i], pa[i + 1], r)
                if d < best:
                    best = d
            if wp.length(pb[i + 1] - pb[i]) < _MAXSEG:
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
            if wp.length(pa[i + 1] - pa[i]) < _MAXSEG:
                d = sd_capsule(p, pa[i], pa[i + 1], r)
                if d < best:
                    best = d
                    seg = i
            if wp.length(pb[i + 1] - pb[i]) < _MAXSEG:
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
         cerode: float, shide: float) -> float:
    d = _MAXD
    if cerode < 1.6:
        d = wp.min(d, board_map(p) + cerode)
    if shide < 1.6:
        d = wp.min(d, _tube(p, pa, pb, rad, n) + shide)
        d = wp.min(d, _rungs(p, ra, rb, rr, nr) + shide)
    return d


@wp.func
def _normal(p: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
            rad: wp.array(dtype=float), n: int,
            ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float), nr: int,
            cerode: float, shide: float) -> wp.vec3:
    e = 0.0015
    dx = _map(p + wp.vec3(e, 0.0, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide) - _map(p - wp.vec3(e, 0.0, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
    dy = _map(p + wp.vec3(0.0, e, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide) - _map(p - wp.vec3(0.0, e, 0.0), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
    dz = _map(p + wp.vec3(0.0, 0.0, e), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide) - _map(p - wp.vec3(0.0, 0.0, e), pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _ao(p: wp.vec3, nrm: wp.vec3, pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
        rad: wp.array(dtype=float), n: int,
        ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float), nr: int,
        cerode: float, shide: float) -> float:
    o = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.05 * float(k)
        d = _map(p + nrm * hr, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
        o += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * o, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), tok: wp.array3d(dtype=wp.int32),
                   nx: int, ny: int, nz: int,
                   pa: wp.array(dtype=wp.vec3), pb: wp.array(dtype=wp.vec3),
                   rad: wp.array(dtype=float), col: wp.array(dtype=wp.vec3), n: int,
                   ra: wp.array(dtype=wp.vec3), rb: wp.array(dtype=wp.vec3), rr: wp.array(dtype=float),
                   rcol: wp.array(dtype=wp.vec3), nr: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, cerode: float, shide: float, tokamt: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(210):
        p = eye + rd * t
        d = _map(p, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
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
    nrm = _normal(p, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
    ao = _ao(p, nrm, pa, pb, rad, n, ra, rb, rr, nr, cerode, shide)
    ldir = wp.normalize(wp.vec3(0.45, 0.8, 0.45))
    lit = wp.clamp(wp.dot(nrm, ldir), 0.0, 1.0)
    hv = wp.normalize(ldir - rd)                              # specular highlight -> rounded look
    spec = wp.pow(wp.clamp(wp.dot(nrm, hv), 0.0, 1.0), 26.0) * 0.5
    rim = wp.pow(1.0 - wp.clamp(-wp.dot(nrm, rd), 0.0, 1.0), 3.0) * 0.18
    shade = (0.12 + 0.95 * lit) * (ao * ao)                  # deep crevice AO -> 3D coils read
    dcard = _MAXD
    if cerode < 1.6:
        dcard = board_map(p) + cerode
    dtube = _MAXD
    drung = _MAXD
    if shide < 1.6:
        dtube = _tube(p, pa, pb, rad, n) + shide
        drung = _rungs(p, ra, rb, rr, nr) + shide

    if drung <= dcard and drung <= dtube:
        bc = rcol[_rungseg(p, ra, rb, rr, nr)]                # a base-pair rung (A/T/G/C)
        img[i, j] = bc * shade + wp.vec3(spec, spec, spec) + bc * (0.10 + rim)
        return
    if dtube <= dcard:
        tc = col[_tubeseg(p, pa, pb, rad, n)]                 # the strand: a card token
        img[i, j] = tc * shade + wp.vec3(spec, spec, spec) + tc * (0.10 + rim)
        return

    # the card surface — painted as ~a million token cells; tokamt fades the card look into the tokens
    bshade = board_shade(p, nrm, rd, ao, time)
    tc = _tokcolor(_voxtok(p, tok, nx, ny, nz))
    toklook = tc * ((0.5 + 0.55 * lit) * ao) * _seam(p)
    img[i, j] = bshade * (1.0 - tokamt) + toklook * tokamt


def _fwd(s):
    """forward half s in [0,1] -> (reveal, cerode, shide, tokamt). STRICT sequence, nothing parallel:
    card -> million tokens -> the chain grows loose->packed (helix -> nucleosomes -> chromosome X)."""
    if s < 0.08:                                    # 1. the real card
        return 0.0, 0.0, 3.0, 0.0
    if s < 0.20:                                    # 2. the card becomes ~a million tokens
        return 0.0, 0.0, 3.0, _ss((s - 0.08) / 0.12)
    if s < 0.28:                                    #    hold the million-token card
        return 0.0, 0.0, 3.0, 1.0
    if s < 0.92:                                    # 3-5. the journey grows: base pairs -> double helix
        r = _ss((s - 0.28) / 0.64)                  #      -> nucleosomes -> chromosome X
        return r, 2.0 * r, 0.0, 1.0
    return 1.0, 2.0, 0.0, 1.0                       # hold the full chain (the whole journey visible)


def _state(time):
    u = (float(time) % _CYCLE) / _CYCLE
    s = u / 0.5 if u < 0.5 else (1.0 - u) / 0.5
    return _fwd(s)


def _render(width, height, time, mouse, device):
    reveal, cerode, shide, tokamt = _state(time)
    posA, posB, radn, raa, rbb, rrr = _positions(reveal)
    pa = wp.array(posA, dtype=wp.vec3, device=device)
    pb = wp.array(posB, dtype=wp.vec3, device=device)
    rad = wp.array(radn, dtype=float, device=device)
    col = wp.array(_COL, dtype=wp.vec3, device=device)
    ra = wp.array(raa, dtype=wp.vec3, device=device)
    rb = wp.array(rbb, dtype=wp.vec3, device=device)
    rr = wp.array(rrr, dtype=float, device=device)
    rcol = wp.array(_RCOL, dtype=wp.vec3, device=device)
    tok = wp.array3d(_TOK3D, dtype=wp.int32, device=device)

    lift = reveal
    az = 0.28 + 0.05 * math.sin(time * 0.10) + float(mouse[0]) * 0.006
    el = 0.42 * (1.0 - lift) + 0.14 * lift
    dist = 9.0 * (1.0 - lift) + 12.2 * lift
    tgt = wp.vec3(-0.1 * (1.0 - lift) + (-2.0) * lift, 0.15 * (1.0 - lift) + (_YC + 1.85) * lift, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(48.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, tok, _NX, _NY, _NZ, pa, pb, rad, col, _N, ra, rb, rr, rcol, _NR,
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(cerode), float(shide), float(tokamt)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_tokenize_chromo",
    description="C3, the whole journey as one chain (nothing skipped or parallel): the real RTX board "
                "becomes ~a million token cells that connect into two backbones with coloured base-pair "
                "rungs, wind into a DNA double helix, bead into nucleosomes, and coil into the two "
                "chromatids of the metaphase chromosome X — every packing level visible at once — then "
                "unwind back to the card.",
    renderer=_render,
)
