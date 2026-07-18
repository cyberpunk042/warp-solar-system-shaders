"""Process 7 scene — the whole thread feeds through the telomere leg and weaves into the chromosome.

The culmination, and the magic. Chains directly from Process 6: it starts from the exact telomere-capped
fibre Process 6 produced — the forest of 30 nm ropes with the bare-DNA **t-loop leg** hanging off the near
end. Then the entire strand is drawn **through that one leg** (the loop is the conduit) and, on the far side,
weaves into a compact chromosome: every base pair streams to the leg in sequence, passes through it, and is
laid onto a tight coil that grows into the condensed, banded, stained-purple metaphase chromosome — a pinched
centromere, rounded telomere-capped tips. The forest empties as the chromosome fills. It is the DNA-compaction
ladder finished by the same combine-the-same-elements idea as the scan-and-merge codec: the thread folds
shoulder-to-shoulder into the smallest, tightest body it can — the ultimate compression — carried entirely by
the real base pairs (nothing spawned, the turns touch but never pass through). A fixed camera watches the
thread pour through the leg and the chromosome bloom.
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

# --- the leg (the telomere loop of the near end 0) is the conduit; the chromosome forms just beyond it ---
_i = np.arange(_P)
_tel_mid = 0.5 * (_CH.tel_a + _CH.tel_b)
_end0 = _CH.is_tel & (_i < _P // 2)
_LEG = _tel_mid[0].astype(np.float32)                                  # the telomere's free 3' TIP — the conduit the whole thread feeds through into the chromosome
_fc = _tel_mid.mean(axis=0)
_out = np.array([_LEG[0] - _fc[0], 0.0, _LEG[2] - _fc[2]], np.float32)
_out = _out / max(float(np.linalg.norm(_out)), 1e-3)                    # forest -> leg, horizontal
_CC = (_LEG + _out * 3.6 + np.array([0.0, -1.4, 0.0], np.float32)).astype(np.float32)   # chromosome centre, beyond the leg

# the folded coil is origin-centred; place it beyond the leg
_chr_a = (_CH.chr_a + _CC).astype(np.float32)
_chr_b = (_CH.chr_b + _CC).astype(np.float32)

# stained-chromosome banding (final purple), stable from the condensed y
_yc = 0.5 * (_CH.chr_a[:, 1] + _CH.chr_b[:, 1])
_yn = _yc / max(float(np.abs(_yc).max()), 1e-6)
_raw = 0.60 * np.sin(np.abs(_yn) * 17.0 + 0.5) + 0.40 * np.sin(np.abs(_yn) * 9.3 + 2.1)
_gb = np.clip((_raw * 0.5 + 0.5 - 0.34) / 0.30, 0.0, 1.0)
_dark = np.array([0.26, 0.16, 0.42], np.float32)
_lite = np.array([0.80, 0.64, 0.90], np.float32)
_chromo = (_dark[None] * (1.0 - _gb[:, None]) + _lite[None] * _gb[:, None]).astype(np.float32)
_greentip = np.array([0.55, 0.92, 0.62], np.float32)
_chromo[_CH.is_tel] = 0.5 * _chromo[_CH.is_tel] + 0.5 * _greentip

_ta = _tb = _ca = _cb = _col = _acol = _bcol = None


def _ensure(device):
    global _ta, _tb, _ca, _cb, _col, _acol, _bcol
    if _ta is None:
        _ta = wp.array(_CH.tel_a, dtype=wp.vec3, device=device)
        _tb = wp.array(_CH.tel_b, dtype=wp.vec3, device=device)
        _ca = wp.array(_chr_a, dtype=wp.vec3, device=device)
        _cb = wp.array(_chr_b, dtype=wp.vec3, device=device)
        _col = wp.array(_chromo, dtype=wp.vec3, device=device)
        _acol = wp.array(_CH.a_col, dtype=wp.vec3, device=device)
        _bcol = wp.array(_CH.b_col, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.46, 0.53, 0.66))     # fibre backbone grey at the start


@wp.func
def _feed(tel: wp.vec3, chrp: wp.vec3, leg: wp.vec3, g: float) -> wp.vec3:
    # g: 0 = still in the forest, 0.5 = passing through the leg, 1 = laid onto the chromosome
    if g <= 0.5:
        return wp.lerp(tel, leg, g * 2.0)
    return wp.lerp(leg, chrp, (g - 0.5) * 2.0)


@wp.kernel
def _weave_kernel(
    ta: wp.array(dtype=wp.vec3),
    tb: wp.array(dtype=wp.vec3),
    ca: wp.array(dtype=wp.vec3),
    cb: wp.array(dtype=wp.vec3),
    col: wp.array(dtype=wp.vec3),
    acol: wp.array(dtype=wp.vec3),
    bcol: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    npairs: int,
    leg: wp.vec3,
    feed: float,            # sweeps -w .. 1+w; pair u is woven when feed passes u
    window: float,
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

    u = float(pr) / float(npairs)
    g = wp.clamp((feed - u) / window + 0.5, 0.0, 1.0)
    gs = g * g * (3.0 - 2.0 * g)

    pa = _feed(ta[pr], ca[pr], leg, g)
    pb = _feed(tb[pr], cb[pr], leg, g)

    # colour greys/greens (fibre + telomere) at the start, condenses to stained purple as it is woven in
    start = _BACKBONE
    if s == 2:
        start = acol[pr]
    if s == 3:
        start = bcol[pr]
    c = wp.lerp(start, col[pr], gs)

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

    # thin bright thread while streaming, fat solid splat once woven into the chromosome body
    base = 0.028 + 0.27 * gs
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
    # feed sweeps the whole strand through the leg over the shot
    return -0.12 + 1.24 * min(max((time - 0.4) / 5.2, 0.0), 1.0)


def _camera(time: float):
    # fixed course: frame the leg + the blooming chromosome, the forest streaming in from the side; ease in.
    u = min(max((time - 0.3) / 5.0, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    target = _CC * (0.5 + 0.5 * u) + _LEG * (0.5 - 0.5 * u)             # pan from the leg onto the chromosome
    perp = np.cross(_out, np.array([0.0, 1.0, 0.0], np.float32))
    perp = perp / max(float(np.linalg.norm(perp)), 1e-3)
    dist = 30.0 * (1.0 - u) + 20.0 * u
    direction = (perp + 0.22 * _out + np.array([0.0, 0.30, 0.0], np.float32)).astype(np.float32)
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
    feed = _schedule(float(time))
    ro, uu, vv, ww, dist = _camera(float(time))
    dnear = float(dist) - 16.0
    dfar = float(dist) + 46.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _weave_kernel,
        dim=_M,
        inputs=[_ta, _tb, _ca, _cb, _col, _acol, _bcol, zbuf, elemcol, W, H, _P,
                wp.vec3(*[float(x) for x in _LEG]), float(feed), 0.30, *cam, 1.7, dnear, dfar],
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
        "Process 7 — the chromosome. The whole telomere-capped fibre from Process 6 is drawn through its "
        "bare-DNA t-loop leg — the loop is the conduit — and woven on the far side into a compact, banded, "
        "stained-purple metaphase chromosome: a pinched centromere, rounded telomere tips, the thread folding "
        "shoulder-to-shoulder into the tightest body it can (the ultimate compression, the same combine-the-"
        "same-elements idea as scan-and-merge). Conserving: chained from Process 6's actual capped fibre, every "
        "base pair streamed through the leg and laid onto the coil, nothing spawned, turns touch but never pass "
        "through — the beautiful chromosome reached through the real fold. The forest empties as it blooms."
    ),
    renderer=_render,
)
