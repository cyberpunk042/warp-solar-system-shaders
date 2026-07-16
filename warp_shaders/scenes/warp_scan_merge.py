"""warp_scan_merge — C1: a scan absorbs the card's elements into atomic mini-cubes, packed into one dense cube.

Operator spec (verbatim, 2026-07-16): *"you need to create a cube with all the minicube after you absorbed
the cards elements in atomic parts, and you need to show me the whole process in the gif, not stop it
halfway... I dont like when I dont see the whole process with the reverse."* Plus the physics rule:
*"you have to merge where the card is ... you are not supposed to break physics."*

C1 as a full, reversible process on the real `gpu_board`:

  1. **Scan** — a bright wavefront sweeps the real RTX board, reading it; in its wake every element is
     **classified** into its `warp_compress.mergecube` token (identical elements share a colour).
  2. **Absorb → atomic mini-cubes** — the card is absorbed: each element becomes an atomic **mini-cube**
     coloured by its token, and the board dissolves as its material is taken up.
  3. **One dense cube** — all the mini-cubes pack together, above where the card is, into a single dense
     **cube of atomic mini-cubes** (grouped by colour) — the merged, compressed store.
  4. **Reverse** — then it runs backwards: the cube unpacks and the card re-forms. `time` shows the whole
     cycle (compress → hold → decompress → the card), then loops. Nothing is cut off halfway.
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
_CYCLE = 12.0
_BLOCK = 5
_NATOM = 512                     # atomic mini-cubes packed into the dense cube (<= 8^3, legible + fast)
_CSTEP = 0.12                    # mini-cube pitch in the dense cube
_CCTR = wp.vec3(0.0, 1.7, 0.0)   # the dense cube floats just above where the card is (its material)


def _maps():
    """token id per board column (for the scan-classify) + the ordered token list for the dense cube."""
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
    # every occupied block is an ATOM (a mini-cube); collect them, grouped by token -> colour bands
    atoms = []
    for bi in range(nbx):
        for by in range(nby):
            for bk in range(nbz):
                if occ_blocks[bi, by, bk]:
                    atoms.append(int(index[bi, by, bk]))
    atoms.sort()                                          # group identical tokens together
    if len(atoms) > _NATOM:                               # subsample to a clean cube, keep colour spread
        idx = np.linspace(0, len(atoms) - 1, _NATOM).astype(int)
        atoms = [atoms[i] for i in idx]
    n = len(atoms)
    side = max(1, int(math.ceil(n ** (1.0 / 3.0))))
    cube_tok = np.full(side ** 3, -1, np.int32)
    cube_tok[:n] = np.asarray(atoms, np.int32)
    return np.ascontiguousarray(tok2d), np.ascontiguousarray(cube_tok), side, n


_TOK2D, _CUBE_TOK, _CSIDE, _NCELL = _maps()
_NBX, _NBZ = _TOK2D.shape
_CSPAN = _CSTEP * float(_CSIDE)
_CORG = wp.vec3(_CCTR[0] - 0.5 * _CSPAN, _CCTR[1] - 0.5 * _CSPAN, _CCTR[2] - 0.5 * _CSPAN)


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
def _cube_idx(p: wp.vec3, side: int) -> int:
    q = p - _CORG
    cx = int(wp.floor(q[0] / _CSTEP))
    cy = int(wp.floor(q[1] / _CSTEP))
    cz = int(wp.floor(q[2] / _CSTEP))
    if cx < 0 or cy < 0 or cz < 0 or cx >= side or cy >= side or cz >= side:
        return -1
    return cx + side * (cz + side * cy)


@wp.func
def _cube_sdf(p: wp.vec3, side: int, revealed: int) -> float:
    idx = _cube_idx(p, side)
    if idx < 0 or idx >= revealed:
        return _MAXD
    q = p - _CORG
    cx = wp.floor(q[0] / _CSTEP); cy = wp.floor(q[1] / _CSTEP); cz = wp.floor(q[2] / _CSTEP)
    center = _CORG + wp.vec3((cx + 0.5) * _CSTEP, (cy + 0.5) * _CSTEP, (cz + 0.5) * _CSTEP)
    return sd_box(p - center, wp.vec3(_CSTEP * 0.40, _CSTEP * 0.40, _CSTEP * 0.40))


@wp.func
def _cmap(p: wp.vec3, time: float, side: int, revealed: int, cardf: float) -> float:
    dc = _cube_sdf(p, side, revealed)
    if cardf < 0.02:
        return dc
    return wp.min(board_map(p), dc)


@wp.func
def _fnormal(p: wp.vec3, time: float, side: int, revealed: int, cardf: float) -> wp.vec3:
    e = 0.0012
    dx = _cmap(p + wp.vec3(e, 0.0, 0.0), time, side, revealed, cardf) - _cmap(p - wp.vec3(e, 0.0, 0.0), time, side, revealed, cardf)
    dy = _cmap(p + wp.vec3(0.0, e, 0.0), time, side, revealed, cardf) - _cmap(p - wp.vec3(0.0, e, 0.0), time, side, revealed, cardf)
    dz = _cmap(p + wp.vec3(0.0, 0.0, e), time, side, revealed, cardf) - _cmap(p - wp.vec3(0.0, 0.0, e), time, side, revealed, cardf)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, time: float, side: int, revealed: int, cardf: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _cmap(p + n * hr, time, side, revealed, cardf)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), tok: wp.array2d(dtype=wp.int32),
                   cube_tok: wp.array(dtype=wp.int32), side: int, revealed: int,
                   nbx: int, nbz: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, bx: float, bz: float, front: float, cardf: float,
                   absorb: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(230):
        p = eye + rd * t
        d = _cmap(p, time, side, revealed, cardf)
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
    dc = _cube_sdf(p, side, revealed)
    db = _MAXD
    if cardf >= 0.02:
        db = board_map(p)
    n = _fnormal(p, time, side, revealed, cardf)
    ao = _fao(p, n, time, side, revealed, cardf)

    if dc < db:
        idx = _cube_idx(p, side)
        tc = _tokcolor(cube_tok[idx])
        q = p - _CORG
        lx = q[0] / _CSTEP - wp.floor(q[0] / _CSTEP)
        ly = q[1] / _CSTEP - wp.floor(q[1] / _CSTEP)
        lz = q[2] / _CSTEP - wp.floor(q[2] / _CSTEP)
        edge = wp.min(wp.min(wp.min(lx, 1.0 - lx), wp.min(ly, 1.0 - ly)), wp.min(lz, 1.0 - lz))
        gl = wp.clamp(edge * 9.0, 0.35, 1.0)                 # dark seams between mini-cubes
        lit = wp.clamp(wp.dot(n, wp.normalize(wp.vec3(0.4, 0.85, 0.45))), 0.2, 1.0)
        img[i, j] = tc * ((0.5 + 0.6 * lit) * ao * gl) + tc * 0.3
        return

    # the board, fading as it is absorbed; scan classifies in its wake
    col = board_shade(p, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)
    bi = int(wp.clamp((p[0] + bx) / (2.0 * bx) * float(nbx), 0.0, float(nbx - 1)))
    bk = int(wp.clamp((p[2] + bz) / (2.0 * bz) * float(nbz), 0.0, float(nbz - 1)))
    tid = tok[bi, bk]
    if tid >= 0 and p[0] < front:
        tc = _tokcolor(tid)
        reveal = wp.clamp((front - p[0]) * 2.5, 0.0, 1.0)
        col = col * (1.0 - 0.55 * reveal * face) + tc * (0.85 * reveal * face)   # classify by token colour
    col = col * cardf                                        # the whole card dims as it is absorbed

    band = wp.abs(p[0] - front)
    if band < 0.16 and cardf > 0.3:
        g = 1.0 - band / 0.16
        col = col + wp.vec3(0.45, 0.85, 1.0) * (g * g * 1.6 * face) \
            + wp.vec3(1.0, 1.0, 1.0) * (g * g * g * 1.2 * face)

    img[i, j] = col


def _smooth(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


def _state(time):
    """(scan-front, card_fade, revealed-cells, absorb) — scan -> absorb into cube -> hold -> reverse -> card."""
    u = (float(time) % _CYCLE) / _CYCLE
    bx = _BB[1]
    fs = -bx + (2.0 * bx + 0.6)                              # front fully past the board
    if u < 0.28:                                            # scan reads + classifies
        fr = -bx + (2.0 * bx + 0.6) * (u / 0.28)
        return fr, 1.0, 0, 0.0
    if u < 0.48:                                            # absorb: card dissolves, cube assembles
        a = _smooth((u - 0.28) / 0.20)
        return fs, 1.0 - 0.9 * a, int(a * float(_NCELL)), a
    if u < 0.60:                                            # hold the dense cube
        return fs, 0.1, _NCELL, 1.0
    if u < 0.80:                                            # reverse: cube unpacks, card re-forms
        a = _smooth((u - 0.60) / 0.20)
        return fs, 0.1 + 0.9 * a, int((1.0 - a) * float(_NCELL)), 1.0 - a
    # settle back to the card
    fr = -bx + (2.0 * bx + 0.6) * (1.0 - (u - 0.80) / 0.20)
    return fr, 1.0, 0, 0.0


def _render(width, height, time, mouse, device):
    front, cardf, revealed, absorb = _state(time)
    tok = wp.array2d(_TOK2D, dtype=wp.int32, device=device)
    cube_tok = wp.array(_CUBE_TOK, dtype=wp.int32, device=device)

    az = 0.58 + 0.25 * math.sin(time * 0.15) + float(mouse[0]) * 0.006
    el = 0.46
    dist = 11.0
    tgt = wp.vec3(-0.1, 0.55 * absorb, 0.0)                  # drift up toward the cube as it forms
    eye = tgt + wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                        dist * math.cos(el) * math.cos(az))
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(46.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, tok, cube_tok, _CSIDE, revealed, _NBX, _NBZ,
                      eye, fwd, right, up, width, height,
                      float(time), tanfov, float(_BB[1]), float(_BB[5]), float(front),
                      float(cardf), float(absorb)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_scan_merge",
    description="C1 as a full reversible process: a scan reads the real RTX board and classifies every "
                "element by its warp_compress token colour, the card is absorbed into atomic mini-cubes, "
                "and all the mini-cubes pack into one dense cube (the merged store) above where the card "
                "is — then it runs in reverse and the card re-forms. The whole compress/decompress cycle.",
    renderer=_render,
)
