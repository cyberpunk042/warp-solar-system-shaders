"""Process 7 scene — the capped fibre folds and condenses into the metaphase chromosome.

The culmination. Chains directly from Process 6: it starts from the exact telomere-capped fibre Process 6
produced and folds it (``fold_chromatid``) into the condensed chromatid — the fibre's centreline wound onto a
short coil with a pinched **centromere** waist and rounded, telomere-capped tips, every base pair carried
rigidly (its own helix/nucleosome/fibre fine structure conserved). Unlike the earlier stages the fibre here
packs **shoulder to shoulder**: a metaphase chromosome IS a dense mass — the turns touch but never pass
through one another (the honest meaning of "condensed"). Rendered as the real folded base pairs (the same
splat style as Processes 3-6), stained the classic banded purple, so the beautiful chromosome is reached
**through** the real compaction, not sculpted from nothing. The camera holds a fixed course (a slow dolly, no
spin) as the whole strand gathers and condenses into the chromosome, in frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome.chromatid import fold_chromatid
from ..scene import Scene

_CH = fold_chromatid(sub=2, block=5)
_P = _CH.n_pairs
_SAMPLES = 4
_M = _P * _SAMPLES

# stained-chromosome banding: a purple ramp modulated by height, stable across the fold (from the final
# condensed y), so the bands read as the chromosome forms.
_yc = 0.5 * (_CH.chr_a[:, 1] + _CH.chr_b[:, 1])
_yn = _yc / max(float(np.abs(_yc).max()), 1e-6)
_raw = 0.60 * np.sin(np.abs(_yn) * 17.0 + 0.5) + 0.40 * np.sin(np.abs(_yn) * 9.3 + 2.1)
_gb = np.clip((_raw * 0.5 + 0.5 - 0.34) / 0.30, 0.0, 1.0)
_dark = np.array([0.26, 0.16, 0.42], np.float32)
_lite = np.array([0.80, 0.64, 0.90], np.float32)
_chromo = (_dark[None] * (1.0 - _gb[:, None]) + _lite[None] * _gb[:, None]).astype(np.float32)
# telomere tips tinted toward their green cap so the two ends read
_chromo[_CH.is_tel] = 0.5 * _chromo[_CH.is_tel] + 0.5 * np.array([0.55, 0.92, 0.62], np.float32)

_ta = _tb = _ca = _cb = _col = None


def _ensure(device):
    global _ta, _tb, _ca, _cb, _col
    if _ta is None:
        _ta = wp.array(_CH.tel_a, dtype=wp.vec3, device=device)
        _tb = wp.array(_CH.tel_b, dtype=wp.vec3, device=device)
        _ca = wp.array(_CH.chr_a, dtype=wp.vec3, device=device)
        _cb = wp.array(_CH.chr_b, dtype=wp.vec3, device=device)
        _col = wp.array(_chromo, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)


@wp.kernel
def _fold_kernel(
    ta: wp.array(dtype=wp.vec3),
    tb: wp.array(dtype=wp.vec3),
    ca: wp.array(dtype=wp.vec3),
    cb: wp.array(dtype=wp.vec3),
    col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    to_chr: float,          # 0 = Process-6 capped fibre, 1 = condensed chromatid
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

    c = col[pr]
    if s == 0:
        pos = pa
    elif s == 1:
        pos = pb
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
    else:
        pos = wp.lerp(pa, pb, 0.64)
    elemcol[e] = c

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

    base = 0.11                                           # fat splats: the condensed chromosome packs solid
    rpx = zoom * base / cz * float(height_px)
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
def _resolve_kernel(
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height_px: int,
):
    i, j = wp.tid()
    yy = float(i) / float(height_px)
    bg = wp.vec3(0.020, 0.018, 0.030) * (1.0 - 0.5 * yy) + wp.vec3(0.006, 0.004, 0.012)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.32 - 1.02 * depthq
    fog = wp.clamp((depthq - 0.30) * 1.6, 0.0, 0.85)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    u = min(max((time - 0.5) / 4.4, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _camera(time: float):
    # fixed course, no spin: dolly in from the wide capped-fibre forest onto the condensed chromosome.
    u = min(max((time - 0.3) / 4.4, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 34.0 * (1.0 - u) + 15.0 * u
    target = np.array([0.0, 0.0, 0.0], np.float32)
    direction = np.array([0.34, 0.16, 1.0], np.float32)
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
    dnear = float(dist) - 10.0
    dfar = float(dist) + 40.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _fold_kernel,
        dim=_M,
        inputs=[_ta, _tb, _ca, _cb, _col, zbuf, elemcol, W, H,
                float(to_chr), *cam, 1.7, dnear, dfar],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.06, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.32)
    return ldr


SCENE = Scene(
    name="warp_chromosome",
    description=(
        "Process 7 — the chromosome. The telomere-capped 30 nm fibre from Process 6 folds and condenses into "
        "the metaphase chromatid: the fibre's centreline winds onto a short coil with a pinched centromere and "
        "rounded telomere-capped tips, every base pair carried rigidly, packing shoulder-to-shoulder into a "
        "dense stained-purple body. Conserving: chained from Process 6's actual capped fibre, nothing spawned, "
        "turns touch but never pass through, fixed camera — the beautiful chromosome reached through the real fold."
    ),
    renderer=_render,
)
