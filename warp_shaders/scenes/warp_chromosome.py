"""Process 6 scene — the 30 nm fibres fold into the metaphase chromosome (the whole ladder's end).

Chains directly from Process 5: it starts from the exact fibre band Process 5 produced and folds it. The
~47 fibres split between the two sister **chromatids**, laid head-to-tail along each bowed arm — telomere
tip → centromere waist → telomere tip — and crushed to fill the condensed rod. The two chromatids meet at
the centromere; the four tips are capped by the telomeres. This is the last ~40× of compaction.

Conserving and physical: every base pair is reused — each fibre is folded (not regenerated) onto its arm,
continuously; no point is created or teleports. The camera holds a fixed course (a slow dolly to present
the chromosome, no spin) as the wide fibre band folds into the chromosome — the whole fold in frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome import fold_chromosome
from ..scene import Scene

_CR = fold_chromosome(sub=2, block=5)
_P = _CR.n_pairs
_SAMPLES = 4
_M = _P * _SAMPLES

_fa = _fb = _ca = _cb = _a_col = _b_col = None


def _ensure(device):
    global _fa, _fb, _ca, _cb, _a_col, _b_col
    if _fa is None:
        _fa = wp.array(_CR.fib_a, dtype=wp.vec3, device=device)
        _fb = wp.array(_CR.fib_b, dtype=wp.vec3, device=device)
        _ca = wp.array(_CR.chr_a, dtype=wp.vec3, device=device)
        _cb = wp.array(_CR.chr_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_CR.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_CR.b_col, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.52, 0.60, 0.74))


@wp.kernel
def _fold_kernel(
    fa: wp.array(dtype=wp.vec3),
    fb: wp.array(dtype=wp.vec3),
    ca: wp.array(dtype=wp.vec3),
    cb: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    to_chr: float,          # 0 = Process-5 fibre band, 1 = folded chromosome
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

    pa = wp.lerp(fa[pr], ca[pr], to_chr)
    pb = wp.lerp(fb[pr], cb[pr], to_chr)

    # base-pair colour; telomere/centromere tints (baked into a_col/b_col) also lift the backbone so the
    # landmarks read as bright caps and a warm waist, not just on the interior points.
    ca_c = a_col[pr]
    cb_c = b_col[pr]
    tinted = ca_c[1] > 0.9

    if s == 0:
        pos = pa
        col = _BACKBONE
        if tinted:
            col = ca_c
    elif s == 1:
        pos = pb
        col = _BACKBONE
        if tinted:
            col = cb_c
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
        col = ca_c
    else:
        pos = wp.lerp(pa, pb, 0.64)
        col = cb_c
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

    base = 0.030
    if s >= 2:
        base = 0.016
    rpx = zoom * base / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 7.0))

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
def _resolve_kernel(
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height_px: int,
):
    i, j = wp.tid()
    yy = float(i) / float(height_px)
    bg = wp.vec3(0.016, 0.020, 0.030) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.30 - 1.05 * depthq
    fog = wp.clamp((depthq - 0.30) * 1.9, 0.0, 0.88)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    u = min(max((time - 0.5) / 4.2, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _camera(time: float):
    # fixed course, no spin: the camera dollies in and swings from looking along the fibre band to facing
    # the chromosome front-on as it folds — presenting the final X, never spinning the subject.
    u = min(max((time - 0.3) / 4.4, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 44.0 * (1.0 - u) + 23.0 * u
    ey = 0.42 * (1.0 - u) + 0.12 * u                     # tilt from down-the-band to near-front-on
    sx = 0.05 * (1.0 - u) + 0.22 * u
    target = np.array([0.0, 0.0, 0.0], np.float32)
    direction = np.array([sx, ey, 1.0], np.float32)
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, dist


def _render(width, height, time, mouse, device):
    _ensure(device)
    W, H = int(width), int(height)
    to_chr = _schedule(float(time))
    ro, uu, vv, ww, dist = _camera(float(time))
    dnear = 1.5
    dfar = float(dist) + 40.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _fold_kernel,
        dim=_M,
        inputs=[_fa, _fb, _ca, _cb, _a_col, _b_col, zbuf, elemcol, W, H,
                float(to_chr), *cam, 1.7, dnear, dfar],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_chromosome",
    description=(
        "Process 6 — the chromosome. The 30 nm fibres from Process 5 fold into the two sister chromatids "
        "of the metaphase chromosome, joined at the centromere and capped by the telomeres — the last of "
        "the packing that took the card's tokens all the way to a chromosome. Conserving: chained from "
        "Process 5's actual fibres, every base pair folded not regenerated, nothing spawned, fixed camera."
    ),
    renderer=_render,
)
