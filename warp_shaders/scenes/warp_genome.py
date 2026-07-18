"""The whole genome compression as ONE continuous scene — the card, retransforming into a chromosome.

A single point cloud of the card's base pairs, morphed **in place** through every stage of the ladder with
**one continuous camera** (it only ever pulls in as the matter compacts — no cuts, no jumps). The same
matter the whole way: the graphics card → its tokens → base pairs → a field of double helices → nucleosome
beads → the 30 nm fibre → the telomere-capped strand → the merged metaphase chromosome. Each keyframe is the
real output of that stage's library (the same pair ordering throughout), so what you watch is one shape
continuously reshaping from board to chromosome.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome.basepair import _disperse_cloud, _morton_order, bind_pairs
from ..genome.chromatid import fold_chromatid
from ..genome.fibre import coil_fibre
from ..genome.helix import wind_helix, wound_positions
from ..genome.nucleosome import wrap_nucleosomes
from ..genome.telomere import cap_telomeres
from ..genome.tokenize import tokenize_card
from ..scene import Scene

_SUB, _BLOCK = 2, 5
_BACKBONE = np.array([0.46, 0.53, 0.66], np.float32)

# stage labels + dwell/morph timing (seconds). one continuous timeline.
_DWELL = 0.7
_MORPH = 2.5
_SEG = _DWELL + _MORPH
_NSEG = 7                       # 8 keyframes, 7 transitions
_T_END = _NSEG * _SEG + 1.2

_built = False
_KA = _KB = _KCA = _KCB = None   # (K,P,3) positions a/b, (K,P,3) colours a/b
_RAD = _CEN = None               # per-keyframe radius + centroid
# target on-screen splat size (px) per keyframe: small for the spread card/field, big to fuse the
# compact merged chromosome's ~143 pieces into a solid body.
_SPX = np.array([3.0, 3.0, 3.2, 3.2, 3.8, 5.0, 5.0, 16.0], np.float32)
_pa = _pb = _ca = _cb = _rad = _cen = None


def _radius_centroid(a, b):
    m = 0.5 * (a + b)
    c = m.mean(0)
    r = np.percentile(np.linalg.norm(m - c, axis=1), 97.0)
    return float(r), c.astype(np.float32)


def _build():
    """Build the 8 keyframes at the pair level, all sharing bind_pairs' pair ordering."""
    global _built, _KA, _KB, _KCA, _KCB, _RAD, _CEN
    tc = tokenize_card(sub=_SUB, block=_BLOCK)
    homes = tc.positions.copy()
    tcol = tc.colors.copy()
    cloud = _disperse_cloud(homes, 1.0)
    if homes.shape[0] % 2:
        homes, cloud, tcol = homes[:-1], cloud[:-1], tcol[:-1]
    order = _morton_order(cloud)
    ai, bi = order[0::2], order[1::2]

    bp = bind_pairs(sub=_SUB, block=_BLOCK)
    hx = wind_helix(sub=_SUB, block=_BLOCK); ha, hb = wound_positions(hx)
    nc = wrap_nucleosomes(sub=_SUB, block=_BLOCK)
    fb = coil_fibre(sub=_SUB, block=_BLOCK)
    tl = cap_telomeres(sub=_SUB, block=_BLOCK)
    ch = fold_chromatid(sub=_SUB, block=_BLOCK)

    # purple banding + green tips for the final chromosome
    yc = 0.5 * (ch.chr_a[:, 1] + ch.chr_b[:, 1]); yn = yc / max(float(np.abs(yc).max()), 1e-6)
    raw = 0.60 * np.sin(np.abs(yn) * 17.0 + 0.5) + 0.40 * np.sin(np.abs(yn) * 9.3 + 2.1)
    gb = np.clip((raw * 0.5 + 0.5 - 0.34) / 0.30, 0.0, 1.0)
    purple = (np.array([0.26, 0.16, 0.42]) * (1 - gb)[:, None] + np.array([0.80, 0.64, 0.90]) * gb[:, None]).astype(np.float32)
    green = np.array([0.55, 0.92, 0.62], np.float32)
    pa_col = purple.copy(); pb_col = purple.copy()
    pa_col[ch.is_tel] = 0.5 * pa_col[ch.is_tel] + 0.5 * green
    pb_col[ch.is_tel] = 0.5 * pb_col[ch.is_tel] + 0.5 * green

    base_a, base_b = bp.a_col, bp.b_col                     # 4-base colours
    tel_a = base_a.copy(); tel_b = base_b.copy()
    tel_a[tl.is_tel] = green; tel_b[tl.is_tel] = green

    # keyframes: (a_pos, b_pos, a_col, b_col)
    frames = [
        (homes[ai], homes[bi], tcol[ai], tcol[bi]),               # 0 card
        (cloud[ai], cloud[bi], tcol[ai], tcol[bi]),               # 1 tokens
        (bp.field_a, bp.field_b, base_a, base_b),                 # 2 base pairs
        (ha, hb, base_a, base_b),                                 # 3 double helices
        (nc.nuc_a, nc.nuc_b, base_a, base_b),                     # 4 nucleosomes
        (fb.fib_a, fb.fib_b, base_a, base_b),                     # 5 30 nm fibre
        (tl.tel_a, tl.tel_b, tel_a, tel_b),                       # 6 telomere
        (ch.chr_a, ch.chr_b, pa_col, pb_col),                     # 7 chromosome (merged)
    ]
    P = frames[0][0].shape[0]
    K = len(frames)
    _KA = np.stack([f[0] for f in frames]).astype(np.float32)
    _KB = np.stack([f[1] for f in frames]).astype(np.float32)
    _KCA = np.stack([f[2] for f in frames]).astype(np.float32)
    _KCB = np.stack([f[3] for f in frames]).astype(np.float32)
    rc = [_radius_centroid(f[0], f[1]) for f in frames]
    _RAD = np.array([r for r, _ in rc], np.float32)
    _CEN = np.stack([c for _, c in rc]).astype(np.float32)
    _built = True
    return P, K


def _ensure(device):
    global _pa, _pb, _ca, _cb, _rad, _cen
    if not _built:
        _build()
    if _pa is None:
        _pa = wp.array(_KA, dtype=wp.vec3, device=device)
        _pb = wp.array(_KB, dtype=wp.vec3, device=device)
        _ca = wp.array(_KCA, dtype=wp.vec3, device=device)
        _cb = wp.array(_KCB, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)


@wp.kernel
def _morph_kernel(
    ka: wp.array2d(dtype=wp.vec3),
    kb: wp.array2d(dtype=wp.vec3),
    kca: wp.array2d(dtype=wp.vec3),
    kcb: wp.array2d(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    k0: int,
    k1: int,
    u: float,               # 0..1 morph between keyframe k0 and k1
    splat: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
    dnear: float,
    dfar: float,
):
    e = wp.tid()
    pr = e / 4
    s = e - pr * 4

    pa = wp.lerp(ka[k0, pr], ka[k1, pr], u)
    pb = wp.lerp(kb[k0, pr], kb[k1, pr], u)
    ca = wp.lerp(kca[k0, pr], kca[k1, pr], u)
    cb = wp.lerp(kcb[k0, pr], kcb[k1, pr], u)

    if s == 0:
        pos = pa; col = ca
    elif s == 1:
        pos = pb; col = cb
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36); col = ca
    else:
        pos = wp.lerp(pa, pb, 0.64); col = cb
    elemcol[e] = col

    rel = pos - ro
    cz = wp.dot(rel, ww)
    if cz < 0.05:
        return
    cx = wp.dot(rel, uu)
    cy = wp.dot(rel, vv)
    pfx = zoom * cx / cz * float(height_px) + 0.5 * float(width) - 0.5
    pfy = 0.5 * float(height_px) - 0.5 - zoom * cy / cz * float(height_px)
    px = int(wp.round(pfx))
    py = int(wp.round(pfy))

    rpx = zoom * splat / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 12.0))

    depthq = int(wp.clamp((cz - dnear) / (dfar - dnear) * 1022.0, 0.0, 1022.0))
    key = (depthq << _IDX_BITS) | e
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if float(dx * dx + dy * dy) <= float(rad * rad) + 0.5:
                xx = px + dx
                yy = py + dy
                if xx >= 0 and xx < width and yy >= 0 and yy < height_px:
                    wp.atomic_min(zbuf, yy, xx, key)


@wp.kernel
def _resolve_kernel(zbuf: wp.array2d(dtype=wp.int32), elemcol: wp.array(dtype=wp.vec3),
                    img: wp.array2d(dtype=wp.vec3), width: int, height_px: int):
    i, j = wp.tid()
    yy = float(i) / float(height_px)
    bg = wp.vec3(0.017, 0.019, 0.030) * (1.0 - 0.5 * yy) + wp.vec3(0.005, 0.004, 0.010)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.30 - 1.05 * depthq
    fog = wp.clamp((depthq - 0.28) * 1.7, 0.0, 0.9)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _phase(time: float):
    """Return (k0, k1, u, radius, centroid) for the continuous timeline with a dwell at each keyframe."""
    t = min(max(float(time), 0.0), _T_END)
    seg = min(int(t // _SEG), _NSEG - 1)
    local = t - seg * _SEG
    u = 0.0 if local <= _DWELL else min((local - _DWELL) / _MORPH, 1.0)
    u = u * u * (3.0 - 2.0 * u)
    k0, k1 = seg, seg + 1
    rad = (1.0 - u) * _RAD[k0] + u * _RAD[k1]
    cen = (1.0 - u) * _CEN[k0] + u * _CEN[k1]
    spx = (1.0 - u) * _SPX[k0] + u * _SPX[k1]
    return k0, k1, float(u), float(rad), cen, float(spx)


def _render(width, height, time, mouse, device):
    _ensure(device)
    W, H = int(width), int(height)
    k0, k1, u, rad, cen, spx = _phase(time)
    P = _KA.shape[1]
    M = P * 4

    target = cen.astype(np.float32)
    dist = 2.7 * rad + 3.0
    splat = spx * dist / (1.7 * float(H))                         # world size for the target on-screen px
    direction = np.array([0.30, 0.32, 1.0], np.float32)
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro; ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32)); uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    dnear = max(dist - 2.2 * rad, 0.4); dfar = dist + 3.0 * rad + 6.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(_morph_kernel, dim=M,
              inputs=[_pa, _pb, _ca, _cb, zbuf, elemcol, W, H, k0, k1, u, float(splat),
                      *cam, 1.7, float(dnear), float(dfar)], device=device)
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.04, preserve_hue=True)
    return post.vignette(ldr, amount=0.3)


SCENE = Scene(
    name="warp_genome",
    description=(
        "The whole genome compression in one continuous take: a single point cloud of the card's base pairs "
        "morphs in place — card -> tokens -> base pairs -> double helices -> nucleosomes -> 30 nm fibre -> "
        "telomere -> merged chromosome — with one camera that only pulls in as the matter compacts. Same "
        "matter throughout, the real output of each stage's library, no cuts."
    ),
    renderer=_render,
)
