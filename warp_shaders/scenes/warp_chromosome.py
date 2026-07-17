"""Process 7 scene — the telomere-capped fibre folds into a single-chromatid chromosome.

Chains directly from Process 6: it starts from the exact telomere-capped strand Process 6 produced and
condenses it into one chromatid — a rounded rod with a **centromere** constriction at the middle and the
two real **telomere** t-loops (green) capping its two ends. One continuous strand, nothing copied — fully
conserving. (The replicated metaphase X is its own scene, ``warp_chromosome_x``.)

Conserving and physical: every base pair is folded (not regenerated) onto the rod; the telomere caps are
carried intact to the tips; no point is created or teleports. The camera holds a fixed 3/4 course (a slow
dolly, no spin) as the fibre folds into the chromosome, whole fold in frame.
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

_ta = _tb = _ca = _cb = _a_col = _b_col = None


def _ensure(device):
    global _ta, _tb, _ca, _cb, _a_col, _b_col
    if _ta is None:
        _ta = wp.array(_CR.tel_a, dtype=wp.vec3, device=device)
        _tb = wp.array(_CR.tel_b, dtype=wp.vec3, device=device)
        _ca = wp.array(_CR.chr_a, dtype=wp.vec3, device=device)
        _cb = wp.array(_CR.chr_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_CR.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_CR.b_col, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.62, 0.68, 0.82))


@wp.kernel
def _fold_kernel(
    ta: wp.array(dtype=wp.vec3),
    tb: wp.array(dtype=wp.vec3),
    ca: wp.array(dtype=wp.vec3),
    cb: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    to_chr: float,
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

    pa = wp.lerp(ta[pr], ca[pr], to_chr)
    pb = wp.lerp(tb[pr], cb[pr], to_chr)

    ac = a_col[pr]
    bc = b_col[pr]
    tel = ac[1] > 0.9                                     # telomere-green pairs lift the backbone too

    if s == 0:
        pos = pa
        col = _BACKBONE
        if tel:
            col = ac
    elif s == 1:
        pos = pb
        col = _BACKBONE
        if tel:
            col = bc
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
        col = ac
    else:
        pos = wp.lerp(pa, pb, 0.64)
        col = bc
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

    base = 0.052                                        # fat splats so the dense volume fill reads as opaque
    if s >= 2:
        base = 0.046
    rpx = zoom * base / cz * float(height_px)
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
    shade = 1.34 - 1.02 * depthq                          # near side bright, far side dark -> round rod
    fog = wp.clamp((depthq - 0.34) * 1.7, 0.0, 0.82)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    u = min(max((time - 0.5) / 4.2, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _camera(time: float):
    # fixed 3/4 course, no spin: dolly in from the capped strand to the folded chromatid, a side-and-above
    # angle so the rod reads as round (not a flat blade).
    u = min(max((time - 0.3) / 4.4, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 40.0 * (1.0 - u) + 24.0 * u
    target = np.array([0.0, 0.7, 0.0], np.float32)
    direction = np.array([0.5, 0.12, 1.0], np.float32)
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
    dnear = float(dist) - 6.0                             # tight range: shade the rod's near/far -> round
    dfar = float(dist) + 6.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _fold_kernel,
        dim=_M,
        inputs=[_ta, _tb, _ca, _cb, _a_col, _b_col, zbuf, elemcol, W, H,
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
        "Process 7 — the chromosome (single chromatid). The telomere-capped fibre from Process 6 condenses "
        "into one chromatid: a rounded rod with a centromere constriction and the two real telomere t-loops "
        "(green) capping its ends. One strand, nothing copied — fully conserving; chained from Process 6's "
        "actual output, every base pair folded not regenerated, nothing spawned, fixed 3/4 camera."
    ),
    renderer=_render,
)
