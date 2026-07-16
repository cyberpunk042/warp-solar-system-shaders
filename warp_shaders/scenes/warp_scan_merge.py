"""warp_scan_merge — C1: a scan reads the card, the same elements merge IN PLACE into growing digit-cubes.

Operator spec (verbatim): *"one compression for example can just merge the same thing together and have
digit to represent the locations of the vsrious same element that also grow a but the cube part."*
And, sharply: *"you have to merge where the card is ... you are not supposed to break physics"* — the
compression must happen **on the card**, not teleport into a fabricated cube floating in empty space.

The per-element merge (C1) as a physically-honest process on the real `gpu_board`:

  1. **Scan** — a bright wavefront sweeps the real RTX board, reading it. In its wake every repeated
     element is **classified**: identical pieces glow the **same colour** (the same C1 token, from the
     card's own block-dedup in `warp_compress.mergecube`).
  2. **Merge, where the card is** — for each repeated element, all its copies **merge into one place on
     the board** — its own canonical location — and a **cube of digits grows right there**, out of the
     card, its size set by **how many copies merged** (the count = the digits that represent the
     locations of the various same element). The redundant copies fade to dim ghosts as their
     information moves into the growing cube. The cube grows **on the card**, never beside it.

Two patches share a hue iff they are byte-identical blocks in `warp_compress.mergecube`; a growing
cube's size is exactly that token's occurrence count. `time` runs scan → merge → hold, then loops.
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
from warp_compress.foldcube import sample_card, _BB

_MAXD = 40.0
_CYCLE = 10.0
_BLOCK = 5
_NTOK = 16                       # the most-repeated elements each grow a cube (kept legible + fast)
_CSTEP = 0.135                   # cube side = _CSTEP * (merged count)^(1/3)
_CBASE = 0.30                    # cube base y — sits on top of the board's components, grows upward
_CMAX = 0.62                     # cap the cube side
_PS = 0.15                       # packed mini-cube side once gathered into the one storage cube
_ACX = -0.10                     # the assembled cube rests on the board, near the card's centre
_ACZ = 0.0


def _maps():
    """token id per board column + canonical mask + per-token (canonical world x,z on the card, count)."""
    b = _BLOCK
    occ = sample_card()
    vocab, index, meta = mc.compress(occ, block=b)
    occp = mc._pad_to(occ, b)
    nbx, nby, nbz = index.shape
    blk = (occp.reshape(nbx, b, nby, b, nbz, b)
               .transpose(0, 2, 4, 1, 3, 5)
               .reshape(nbx, nby, nbz, b ** 3))
    occ_blocks = blk.any(axis=3)
    tok2d = np.full((nbx, nbz), -1, np.int32)
    for by in range(nby - 1, -1, -1):
        has = occ_blocks[:, by, :]
        fill = (tok2d < 0) & has
        tok2d[fill] = index[:, by, :][fill]
    ids, counts = np.unique(tok2d[tok2d >= 0], return_counts=True)
    cmap = {int(i): int(c) for i, c in zip(ids, counts)}
    repeated = set(i for i in cmap if cmap[i] >= 4)          # "the same thing" in many places
    canon = np.zeros((nbx, nbz), np.int32)
    seen = set()
    pos = {}
    for bi in range(nbx):
        for bk in range(nbz):
            tid = int(tok2d[bi, bk])
            if tid not in repeated:
                tok2d[bi, bk] = -1                            # unique/rare pieces stay bare board
                continue
            if tid not in seen:
                seen.add(tid)
                canon[bi, bk] = 1                            # one canonical survivor per element
                pos[tid] = (bi, bk)
    # top-N repeated elements by count — each grows a cube at ITS OWN spot on the card
    order = sorted(repeated, key=lambda t: -cmap[t])[:_NTOK]
    bx, bz = _BB[1], _BB[5]
    n = len(order)
    tcx = np.zeros(n, np.float32); tcz = np.zeros(n, np.float32)
    tcnt = np.zeros(n, np.float32); ttid = np.zeros(n, np.int32)
    for k, tid in enumerate(order):
        bi, bk = pos[tid]
        tcx[k] = -bx + (bi + 0.5) / nbx * 2.0 * bx           # canonical location, in board/world space
        tcz[k] = -bz + (bk + 0.5) / nbz * 2.0 * bz
        tcnt[k] = float(cmap[tid])
        ttid[k] = tid
    # a slot per mini-cube in the ONE assembled storage cube — a tight k-grid on the board.
    m = int(math.ceil(n ** (1.0 / 3.0))) if n > 0 else 1
    half = (m - 1) * 0.5
    slx = np.zeros(n, np.float32); sly = np.zeros(n, np.float32); slz = np.zeros(n, np.float32)
    for k in range(n):
        kx = k % m; ky = (k // m) % m; kz = k // (m * m)
        slx[k] = _ACX + (kx - half) * _PS
        sly[k] = _CBASE + (ky + 0.5) * _PS                  # ky=0 rests on the board plane
        slz[k] = _ACZ + (kz - half) * _PS
    return (np.ascontiguousarray(tok2d), np.ascontiguousarray(canon),
            np.ascontiguousarray(tcx), np.ascontiguousarray(tcz),
            np.ascontiguousarray(tcnt), np.ascontiguousarray(ttid),
            np.ascontiguousarray(slx), np.ascontiguousarray(sly), np.ascontiguousarray(slz))


_TOK2D, _CANON, _TCX, _TCZ, _TCNT, _TTID, _SLX, _SLY, _SLZ = _maps()
_NBX, _NBZ = _TOK2D.shape
_NT = int(_TCX.shape[0])


@wp.func
def _hue(h: float) -> wp.vec3:
    r = wp.clamp(wp.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = wp.clamp(2.0 - wp.abs(h * 6.0 - 2.0), 0.0, 1.0)
    b = wp.clamp(2.0 - wp.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return wp.vec3(r, g, b)


@wp.func
def _tokcolor(tid: int) -> wp.vec3:
    h = (float(tid) * 0.61803) % 1.0
    return _hue(h)


@wp.func
def _cube_side(cnt: float, mrg: float) -> float:
    revealed = wp.max(mrg * cnt, 0.001)                      # copies merged so far
    return wp.min(_CSTEP * wp.pow(revealed, 0.3333), _CMAX)


@wp.func
def _cube_ctr(k: int, cx: wp.array(dtype=float), cz: wp.array(dtype=float),
              cnt: wp.array(dtype=float), slx: wp.array(dtype=float), sly: wp.array(dtype=float),
              slz: wp.array(dtype=float), mrg: float, con: float) -> wp.vec3:
    s0 = _cube_side(cnt[k], mrg)
    c0 = wp.vec3(cx[k], _CBASE + s0 * 0.5, cz[k])            # grown in place, on the card
    c1 = wp.vec3(slx[k], sly[k], slz[k])                    # its slot in the one storage cube
    return c0 * (1.0 - con) + c1 * con                      # slides continuously — no teleport


@wp.func
def _cube_hs(k: int, cnt: wp.array(dtype=float), mrg: float, con: float) -> float:
    s0 = _cube_side(cnt[k], mrg)
    return (s0 * (1.0 - con) + _PS * con) * 0.5             # half-side, lerped to the packed size


@wp.func
def _cube_sdf(p: wp.vec3, nt: int, cx: wp.array(dtype=float), cz: wp.array(dtype=float),
              cnt: wp.array(dtype=float), slx: wp.array(dtype=float), sly: wp.array(dtype=float),
              slz: wp.array(dtype=float), mrg: float, con: float) -> float:
    best = _MAXD
    for k in range(nt):
        c = _cube_ctr(k, cx, cz, cnt, slx, sly, slz, mrg, con)
        h = _cube_hs(k, cnt, mrg, con)
        d = sd_box(p - c, wp.vec3(h, h, h))
        if d < best:
            best = d
    return best


@wp.func
def _cube_nearest(p: wp.vec3, nt: int, cx: wp.array(dtype=float), cz: wp.array(dtype=float),
                  cnt: wp.array(dtype=float), slx: wp.array(dtype=float), sly: wp.array(dtype=float),
                  slz: wp.array(dtype=float), mrg: float, con: float) -> int:
    best = _MAXD
    idx = int(-1)
    for k in range(nt):
        c = _cube_ctr(k, cx, cz, cnt, slx, sly, slz, mrg, con)
        h = _cube_hs(k, cnt, mrg, con)
        d = sd_box(p - c, wp.vec3(h, h, h))
        if d < best:
            best = d
            idx = k
    return idx


@wp.func
def _cmap(p: wp.vec3, nt: int, cx: wp.array(dtype=float), cz: wp.array(dtype=float),
          cnt: wp.array(dtype=float), slx: wp.array(dtype=float), sly: wp.array(dtype=float),
          slz: wp.array(dtype=float), mrg: float, con: float, berode: float) -> float:
    # +berode erodes the card away as the cube gathers: when the storage cube is done, the card is GONE
    return wp.min(board_map(p) + berode, _cube_sdf(p, nt, cx, cz, cnt, slx, sly, slz, mrg, con))


@wp.func
def _fnormal(p: wp.vec3, nt: int, cx: wp.array(dtype=float), cz: wp.array(dtype=float),
             cnt: wp.array(dtype=float), slx: wp.array(dtype=float), sly: wp.array(dtype=float),
             slz: wp.array(dtype=float), mrg: float, con: float, berode: float) -> wp.vec3:
    e = 0.0012
    dx = _cmap(p + wp.vec3(e, 0.0, 0.0), nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode) - _cmap(p - wp.vec3(e, 0.0, 0.0), nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode)
    dy = _cmap(p + wp.vec3(0.0, e, 0.0), nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode) - _cmap(p - wp.vec3(0.0, e, 0.0), nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode)
    dz = _cmap(p + wp.vec3(0.0, 0.0, e), nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode) - _cmap(p - wp.vec3(0.0, 0.0, e), nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, nt: int, cx: wp.array(dtype=float), cz: wp.array(dtype=float),
         cnt: wp.array(dtype=float), slx: wp.array(dtype=float), sly: wp.array(dtype=float),
         slz: wp.array(dtype=float), mrg: float, con: float, berode: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _cmap(p + n * hr, nt, cx, cz, cnt, slx, sly, slz, mrg, con, berode)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), tok: wp.array2d(dtype=wp.int32),
                   canon: wp.array2d(dtype=wp.int32),
                   tcx: wp.array(dtype=float), tcz: wp.array(dtype=float),
                   tcnt: wp.array(dtype=float), ttid: wp.array(dtype=wp.int32),
                   slx: wp.array(dtype=float), sly: wp.array(dtype=float), slz: wp.array(dtype=float),
                   nt: int, nbx: int, nbz: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, bx: float, bz: float, front: float, mrg: float,
                   con: float, berode: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(220):
        p = eye + rd * t
        d = _cmap(p, nt, tcx, tcz, tcnt, slx, sly, slz, mrg, con, berode)
        if d < 0.0007 * t + 0.0004:
            hit = 1
            break
        t += d * 0.8
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = ec.studio_sky(rd)
        return

    p = eye + rd * t
    db = board_map(p) + berode
    dc = _cube_sdf(p, nt, tcx, tcz, tcnt, slx, sly, slz, mrg, con)
    n = _fnormal(p, nt, tcx, tcz, tcnt, slx, sly, slz, mrg, con, berode)
    ao = _fao(p, n, nt, tcx, tcz, tcnt, slx, sly, slz, mrg, con, berode)

    if dc < db:
        # a digit-cube: grown in place on the card, then gathered into the one storage cube
        idx = _cube_nearest(p, nt, tcx, tcz, tcnt, slx, sly, slz, mrg, con)
        tc = _tokcolor(ttid[idx])
        c = _cube_ctr(idx, tcx, tcz, tcnt, slx, sly, slz, mrg, con)
        h = _cube_hs(idx, tcnt, mrg, con)
        loc = (p - c) / wp.max(2.0 * h, 0.001)               # -0.5..0.5 within the cube
        # a lattice of digit-cells: bright faces, dark grid lines
        gx = wp.abs(wp.sin(loc[0] * 22.0))
        gy = wp.abs(wp.sin(loc[1] * 22.0))
        gz = wp.abs(wp.sin(loc[2] * 22.0))
        grid = wp.min(wp.min(gx, gy), gz)
        lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.45))), 0.2, 1.0)
        img[i, j] = tc * ((0.5 + 0.7 * lit) * ao) * (0.55 + 0.45 * grid) + tc * 0.35
        return

    # the board itself — scan classify + duplicates ghosting into their cube
    col = board_shade(p, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)
    bi = int(wp.clamp((p[0] + bx) / (2.0 * bx) * float(nbx), 0.0, float(nbx - 1)))
    bk = int(wp.clamp((p[2] + bz) / (2.0 * bz) * float(nbz), 0.0, float(nbz - 1)))
    tid = tok[bi, bk]
    if tid >= 0 and p[0] < front:
        tc = _tokcolor(tid)
        reveal = wp.clamp((front - p[0]) * 2.5, 0.0, 1.0)
        is_canon = float(canon[bi, bk])
        keep = 1.0 - mrg * (1.0 - is_canon)
        amt = reveal * keep
        col = col * (1.0 - 0.6 * amt * face) + tc * (0.9 * amt * face)
        # merged duplicates fade to a dim ghost (their info moved into the growing cube on the card)
        gone = mrg * (1.0 - is_canon) * reveal
        col = col * (1.0 - 0.62 * gone * face) + tc * (0.10 * gone * face)

    # the scan wavefront: a bright cyan-white bar
    band = wp.abs(p[0] - front)
    if band < 0.18:
        g = 1.0 - band / 0.18
        col = col + wp.vec3(0.45, 0.85, 1.0) * (g * g * 1.7 * face) \
            + wp.vec3(1.0, 1.0, 1.0) * (g * g * g * 1.3 * face)

    img[i, j] = col


def _smooth(a, b, x):
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _state(time):
    # One cycle is the WHOLE process AND its reverse: p runs 0 -> 1 -> 0 (compress, then decompress).
    ph = (float(time) % _CYCLE) / _CYCLE * 2.0
    p = ph if ph <= 1.0 else 2.0 - ph
    bx = _BB[1]
    front = -bx + (2.0 * bx + 0.6) * min(p / 0.40, 1.0)      # scan sweeps across, reading + classifying
    mrg = _smooth(0.30, 0.52, p)                             # copies merge in place; digit-cubes grow
    con = _smooth(0.58, 0.82, p)                             # the mini-cubes gather into ONE storage cube
    berode = 1.6 * _smooth(0.60, 0.86, p)                    # the card ERODES away as they gather — gone
    return front, mrg, con, berode                          # by the time the cube is done, no card left


def _render(width, height, time, mouse, device):
    front, mrg, con, berode = _state(time)
    tok = wp.array2d(_TOK2D, dtype=wp.int32, device=device)
    canon = wp.array2d(_CANON, dtype=wp.int32, device=device)
    tcx = wp.array(_TCX, dtype=float, device=device)
    tcz = wp.array(_TCZ, dtype=float, device=device)
    tcnt = wp.array(_TCNT, dtype=float, device=device)
    ttid = wp.array(_TTID, dtype=wp.int32, device=device)
    slx = wp.array(_SLX, dtype=float, device=device)
    sly = wp.array(_SLY, dtype=float, device=device)
    slz = wp.array(_SLZ, dtype=float, device=device)

    az = 0.58 + float(mouse[0]) * 0.006
    el = 0.60
    dist = 9.6 * (1.0 - con) + 3.9 * con                     # dolly in as the storage cube forms
    tgt = wp.vec3(-0.1, 0.25 * (1.0 - con) + (_CBASE + 0.24) * con, 0.0)
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, tok, canon, tcx, tcz, tcnt, ttid, slx, sly, slz, _NT, _NBX, _NBZ,
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(_BB[1]), float(_BB[5]),
                      float(front), float(mrg), float(con), float(berode)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_scan_merge",
    description="C1 as a physically-honest process: a scan sweeps the real RTX board and classifies "
                "every element (identical pieces glow the same warp_compress.mergecube token colour); "
                "then each repeated element's copies merge IN PLACE at its own location on the card, "
                "growing a digit-cube whose size is how many copies merged — the compression happens "
                "where the card is, never in a floating cube beside it.",
    renderer=_render,
)
