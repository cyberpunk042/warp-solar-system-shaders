"""The genome as one continuous **compression** — the DNA thread coils tighter and tighter, weaving into each
denser form, from the extended strand all the way down to the packed chromatid. ONE animation, ONE thread.

This is not a morph (no lerping between poses, nothing flashing into existence). It is the real hierarchical
**coiling** that compacts DNA ~10 000×: a single continuous thread of the card's 182 872 base pairs, whose
shape is a **nested supercoil** — the double helix wound around a nucleosome-scale coil, wound around a
fibre-scale coil, wound around the chromatid axis. A single **condensation** parameter drives the whole
thing: as it rises, the outer axis **shortens** (that shortening *is* the compression) while each level of
coil **engages** in turn (fine → coarse), so the thread weaves ever tighter — every frame a valid,
partially-coiled physical state — until it is a dense, opaque chromatid with a centromere waist and its two
telomere ends. The matter is conserved throughout: the same thread, only ever coiled tighter, never copied
(a copy would be replication → the X, which we do not do), never spawned.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome.basepair import bind_pairs
from ..genome.telomere import cap_telomeres
from ..scene import Scene

_bp = bind_pairs(sub=2, block=5)
_N = int(_bp.n_pairs)
_tl = cap_telomeres(sub=2, block=5)

_A_COL = _bp.a_col.astype(np.float32).copy()
_B_COL = _bp.b_col.astype(np.float32).copy()
_GREEN = np.array([0.45, 1.0, 0.60], np.float32)
_A_COL[_tl.is_tel] = _GREEN                              # the telomere ends, tinted, carried all the way through
_B_COL[_tl.is_tel] = _GREEN
_U = (np.arange(_N) / (_N - 1.0)).astype(np.float64)     # arc position along the one continuous thread

_M = 2 * _N                                              # two backbone strands


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _eng(c, a, b):
    return _smooth((c - a) / (b - a))


def _positions(c):
    """The nested supercoil at condensation ``c`` in [0,1]. Returns the two backbone strands (2N,3) and the
    thread's current half-height (for framing). At c=0 the thread is long and barely coiled; at c=1 it is a
    short, tightly super-coiled, dense rod."""
    u = _U
    L = (1.0 - c) * 62.0 + c * 6.4                       # axis length: long → short  (this shortening = compression)
    waist = 1.0 - 0.42 * c * np.exp(-((u - 0.5) / 0.07) ** 2)          # centromere constriction grows in with c
    arms = 0.35 + 0.65 * np.sqrt(np.clip(1.0 - (2.0 * u - 1.0) ** 8, 0.0, 1.0))  # rounded chromatid arms at high c
    env = (1.0 - c) + c * (arms * waist)                 # radius envelope: uniform when loose, chromatid when dense

    e_fib = _eng(c, 0.30, 0.64)                          # coarse fibre supercoil engages last
    e_nuc = _eng(c, 0.12, 0.42)                          # nucleosome coil engages in the middle
    e_dna = 0.40 + 0.60 * _eng(c, 0.02, 0.16)            # the double helix — mostly always on

    axis = np.zeros((_N, 3))
    axis[:, 1] = (u - 0.5) * L

    # level 1 — the fibre supercoil around the y axis (with an analytic Frenet frame so finer coils nest on it)
    R1 = 1.18 * e_fib * env
    t1 = 46.0
    ph1 = 2.0 * np.pi * t1 * u
    c1, s1 = np.cos(ph1), np.sin(ph1)
    er1 = np.stack([c1, np.zeros_like(c1), s1], 1)
    ep1 = np.stack([-s1, np.zeros_like(s1), c1], 1)
    P1 = axis + R1[:, None] * er1
    dP1 = np.zeros((_N, 3))
    dP1[:, 1] = L
    dP1 += (R1 * 2.0 * np.pi * t1)[:, None] * ep1
    T1 = dP1 / np.linalg.norm(dP1, axis=1, keepdims=True)
    N1 = -er1
    B1 = np.cross(T1, N1)
    B1 /= np.linalg.norm(B1, axis=1, keepdims=True)

    # level 2 — the nucleosome coil wound around the fibre thread (in its N1,B1 frame)
    R2 = 0.42 * e_nuc * env
    t2 = 190.0
    ph2 = 2.0 * np.pi * t2 * u
    P2 = P1 + R2[:, None] * (np.cos(ph2)[:, None] * N1 + np.sin(ph2)[:, None] * B1)

    # level 3 — the DNA double helix wound around the nucleosome thread (two strands, π apart)
    R3 = 0.11 * e_dna * (0.5 + 0.5 * env)
    t3 = 820.0
    ph3 = 2.0 * np.pi * t3 * u
    off_a = R3[:, None] * (np.cos(ph3)[:, None] * N1 + np.sin(ph3)[:, None] * B1)
    off_b = R3[:, None] * (np.cos(ph3 + np.pi)[:, None] * N1 + np.sin(ph3 + np.pi)[:, None] * B1)
    a = (P2 + off_a).astype(np.float32)
    b = (P2 + off_b).astype(np.float32)
    return a, b, float(0.5 * L + 1.3)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)


@wp.kernel
def _splat(
    pa: wp.array(dtype=wp.vec3),
    pb: wp.array(dtype=wp.vec3),
    cola: wp.array(dtype=wp.vec3),
    colb: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    n: int,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
    dnear: float,
    dfar: float,
    base_pt: float,
):
    e = wp.tid()
    if e < n:
        pos = pa[e]
        col = cola[e]
    else:
        pos = pb[e - n]
        col = colb[e - n]
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
    rpx = zoom * base_pt / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 9.0))
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
def _resolve(zbuf: wp.array2d(dtype=wp.int32), elemcol: wp.array(dtype=wp.vec3),
             img: wp.array2d(dtype=wp.vec3), width: int, height_px: int):
    i, j = wp.tid()
    yy = float(i) / float(height_px)
    bg = wp.vec3(0.015, 0.019, 0.029) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.34 - 1.04 * depthq
    fog = wp.clamp((depthq - 0.34) * 1.6, 0.0, 0.85)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


_cola = _colb = None


def _condensation(time: float) -> float:
    return _smooth((float(time) - 0.4) / 9.2)            # slow: the whole compression over ~9.5 s of animation


def _render(width, height, time, mouse, device):
    global _cola, _colb
    W, H = int(width), int(height)
    c = _condensation(float(time))
    pa_np, pb_np, half = _positions(c)

    if _cola is None:
        _cola = wp.array(_A_COL, dtype=wp.vec3, device=device)
        _colb = wp.array(_B_COL, dtype=wp.vec3, device=device)
    pa = wp.array(pa_np, dtype=wp.vec3, device=device)
    pb = wp.array(pb_np, dtype=wp.vec3, device=device)

    # frame the thread: pull in as it compresses (half-height shrinks 32 → ~4.5)
    dist = 2.7 * half + 6.0
    target = np.array([0.0, 0.0, 0.0], np.float32)
    direction = np.array([0.42, 0.16, 1.0], np.float32)
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    dnear = float(dist) - float(half) * 1.1
    dfar = float(dist) + float(half) * 1.1
    base_pt = 0.004 * half + 0.020                        # scales with framed size; dense fill → opaque when packed

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(_splat, dim=_M, inputs=[pa, pb, _cola, _colb, zbuf, elemcol, W, H, _N, *cam, 1.7,
                                      dnear, dfar, float(base_pt)], device=device)
    wp.launch(_resolve, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_genome",
    description=(
        "The genome as one continuous compression — the DNA thread coils tighter and tighter, weaving into "
        "each denser form, from the extended strand down to the packed chromatid. One thread of the card's "
        "182 872 base pairs, shaped as a nested supercoil (double helix → nucleosome coil → fibre coil → "
        "chromatid); a single condensation parameter shortens the axis (that shortening IS the compression) "
        "while each level of coil engages in turn, so the thread weaves ever tighter — every frame a real "
        "partially-coiled state — into a dense, opaque chromatid. Matter conserved, never copied, never "
        "flashed into existence: the same thread, only ever coiled tighter."
    ),
    renderer=_render,
)
