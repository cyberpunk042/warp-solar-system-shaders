"""warp_scan_merge — C1 seen properly: a scan wave reads the RTX card, then duplicates merge to one.

The per-element merge (C1) as a watchable process on the real board:

  1. **Scan** — a bright wavefront sweeps across the real `gpu_board` (in the spirit of the EM wave),
     reading it. In its wake every element is **classified**: identical pieces light the **same colour**
     (the same C1 token, from the card's own block-dedup) — the die, the GDDR7 packages, the VRM chokes,
     the PCB each resolve into their own hue.
  2. **Merge** — the duplicates then **collapse to one**: for every colour, a single canonical element
     stays bright while all its copies fade away — "merge the same thing together" — and the surviving
     representatives pulse. What is kept is one copy per unique element plus, implicitly, where each copy
     was (the location index — the growing "cube part").

The colouring is real: two patches share a hue iff they are byte-identical blocks in
`warp_compress.mergecube`. `time` runs scan → merge → hold, then loops.
"""

import math

import numpy as np
import warp as wp

from .. import electronics_common as ec
from ..engine import post
from ..scene import Scene
from .gpu_board import board_map, board_shade
from warp_compress import mergecube as mc
from warp_compress.foldcube import sample_card, _BB

_MAXD = 40.0
_CYCLE = 10.0
_BLOCK = 5


def _maps():
    """token id per board column + a canonical-instance mask (one representative per token)."""
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
    # keep only the REPEATED elements — "the same thing" that occurs in many places is what merges
    ids, counts = np.unique(tok2d[tok2d >= 0], return_counts=True)
    repeated = set(int(i) for i, c in zip(ids, counts) if c >= 4)
    canon = np.zeros((nbx, nbz), np.int32)
    seen = set()
    for bi in range(nbx):
        for bk in range(nbz):
            tid = int(tok2d[bi, bk])
            if tid not in repeated:
                tok2d[bi, bk] = -1                       # unique/rare pieces stay as bare board
                continue
            if tid not in seen:
                seen.add(tid)
                canon[bi, bk] = 1                        # one canonical survivor per repeated element
    return np.ascontiguousarray(tok2d), np.ascontiguousarray(canon)


_TOK2D, _CANON = _maps()
_NBX, _NBZ = _TOK2D.shape


@wp.func
def _hue(h: float) -> wp.vec3:
    r = wp.clamp(wp.abs(h * 6.0 - 3.0) - 1.0, 0.0, 1.0)
    g = wp.clamp(2.0 - wp.abs(h * 6.0 - 2.0), 0.0, 1.0)
    b = wp.clamp(2.0 - wp.abs(h * 6.0 - 4.0), 0.0, 1.0)
    return wp.vec3(r, g, b)


@wp.func
def _tokcolor(tid: int) -> wp.vec3:
    h = (float(tid) * 0.61803) % 1.0                 # golden-ratio hue -> distinct per token
    return _hue(h)


@wp.func
def _spin(p: wp.vec3, time: float) -> wp.vec3:
    a = 0.08 * wp.sin(time * 0.18)
    ca = wp.cos(a); sa = wp.sin(a)
    return wp.vec3(ca * p[0] + sa * p[2], p[1], -sa * p[0] + ca * p[2])


@wp.func
def _fmap(p: wp.vec3, time: float) -> float:
    return board_map(_spin(p, time))


@wp.func
def _fnormal(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.0012
    dx = _fmap(p + wp.vec3(e, 0.0, 0.0), time) - _fmap(p - wp.vec3(e, 0.0, 0.0), time)
    dy = _fmap(p + wp.vec3(0.0, e, 0.0), time) - _fmap(p - wp.vec3(0.0, e, 0.0), time)
    dz = _fmap(p + wp.vec3(0.0, 0.0, e), time) - _fmap(p - wp.vec3(0.0, 0.0, e), time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _fao(p: wp.vec3, n: wp.vec3, time: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.012 + 0.06 * float(k)
        d = _fmap(p + n * hr, time)
        occ += (hr - d) * sca
        sca *= 0.85
    return wp.clamp(1.0 - 2.0 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), tok: wp.array2d(dtype=wp.int32),
                   canon: wp.array2d(dtype=wp.int32), nbx: int, nbz: int,
                   eye: wp.vec3, fwd: wp.vec3, right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float, bx: float, bz: float, front: float, mrg: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(240):
        p = eye + rd * t
        d = _fmap(p, time)
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
    sp = _spin(p, time)
    n = _fnormal(p, time)
    ao = _fao(p, n, time)
    col = board_shade(sp, n, rd, ao, time)
    face = wp.clamp(n[1], 0.0, 1.0)

    bi = int(wp.clamp((sp[0] + bx) / (2.0 * bx) * float(nbx), 0.0, float(nbx - 1)))
    bk = int(wp.clamp((sp[2] + bz) / (2.0 * bz) * float(nbz), 0.0, float(nbz - 1)))
    tid = tok[bi, bk]

    if tid >= 0 and sp[0] < front:
        tc = _tokcolor(tid)
        reveal = wp.clamp((front - sp[0]) * 2.5, 0.0, 1.0)
        is_canon = float(canon[bi, bk])
        # merge: non-canonical duplicates fade fully back to bare board; one canonical per token stays
        keep = 1.0 - mrg * (1.0 - is_canon)
        amt = reveal * keep
        col = col * (1.0 - 0.6 * amt * face) + tc * (0.9 * amt * face)
        # the surviving copy brightens + pulses as it swallows its duplicates
        surv = mrg * is_canon * reveal
        pulse = 0.6 + 0.4 * wp.sin(float(tid) * 1.7 + time * 5.0)
        col = col + tc * (surv * 1.3 * pulse * face) + wp.vec3(1.0, 1.0, 1.0) * (surv * surv * 0.7 * face)

    # the scan wavefront: a bright cyan-white bar with a soft leading ripple
    band = wp.abs(sp[0] - front)
    if band < 0.18:
        g = 1.0 - band / 0.18
        col = col + wp.vec3(0.45, 0.85, 1.0) * (g * g * 1.7 * face) \
            + wp.vec3(1.0, 1.0, 1.0) * (g * g * g * 1.3 * face)

    img[i, j] = col


def _state(time):
    u = (float(time) % _CYCLE) / _CYCLE
    bx = _BB[1]
    if u < 0.5:                                    # scan sweeps across (read + classify)
        return -bx + (2.0 * bx + 0.5) * (u / 0.5), 0.0
    if u < 0.8:                                    # duplicates merge to one
        return bx + 0.5, (u - 0.5) / 0.3
    return bx + 0.5, 1.0                            # hold the merged state


def _render(width, height, time, mouse, device):
    front, mrg = _state(time)
    tok = wp.array2d(_TOK2D, dtype=wp.int32, device=device)
    canon = wp.array2d(_CANON, dtype=wp.int32, device=device)

    az = 0.52 + float(mouse[0]) * 0.006
    el = 0.72
    dist = 8.7
    eye = wp.vec3(dist * math.cos(el) * math.sin(az), dist * math.sin(el),
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(-0.1, -0.12, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, tok, canon, _NBX, _NBZ, eye, fwd, right, up, width, height,
                      float(time), tanfov, float(_BB[1]), float(_BB[5]), float(front), float(mrg)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="warp_scan_merge",
    description="C1 as a process: a scan wave sweeps the real RTX board and classifies every element "
                "(identical pieces glow the same colour = the same warp_compress.mergecube token), then "
                "the duplicates merge to one canonical copy each. The scan is the read; the merge is the "
                "compress.",
    renderer=_render,
)
