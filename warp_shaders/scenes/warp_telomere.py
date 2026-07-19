"""Process 6 scene — the DNA strand's ends curl into telomere t-loops.

Chains directly from Process 5: it starts from the exact fibre Process 5 produced and curls its two ends.
A linear strand has exactly two ends, so there are exactly two telomeres; the terminal stretch at each end
(tinted telomere-green) leaves the fibre, arcs a lasso, and the free 3' tip tucks back into the duplex — a
**t-loop**, the protective cap. The camera features the near end's t-loop, the fibre trailing away, with
the far end's cap visible down the strand.

Conserving and physical: only the terminal base pairs are reshaped — the strand curls back on itself; no
point is created or teleports. The camera holds a fixed course (a slow dolly, no spin) as the ends curl.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome.telomere import cap_telomeres
from ..scene import Scene

_TL = cap_telomeres(sub=2, block=5)
_P = _TL.n_pairs
_SAMPLES = 4
_M = _P * _SAMPLES
_END = _TL.ends[0]                                        # feature the near end's t-loop
_LOOPC = 0.5 * (_TL.tel_a[:_TL.tel_len] + _TL.tel_b[:_TL.tel_len]).mean(axis=0)   # its lasso centroid
_OUT = np.array([_LOOPC[0], 0.0, _LOOPC[2]], np.float32)
_OUT = _OUT / max(float(np.linalg.norm(_OUT)), 1e-3)     # horizontal outward from the forest centre
_SIDE = np.cross(_OUT, np.array([0.0, 1.0, 0.0], np.float32))
_SIDE = _SIDE / max(float(np.linalg.norm(_SIDE)), 1e-3)  # horizontal, perpendicular to the loop plane

_fa = _fb = _ta = _tb = _a_col = _b_col = None


def _ensure(device):
    global _fa, _fb, _ta, _tb, _a_col, _b_col
    if _fa is None:
        _fa = wp.array(_TL.fib_a, dtype=wp.vec3, device=device)
        _fb = wp.array(_TL.fib_b, dtype=wp.vec3, device=device)
        _ta = wp.array(_TL.tel_a, dtype=wp.vec3, device=device)
        _tb = wp.array(_TL.tel_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_TL.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_TL.b_col, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.46, 0.53, 0.66))


@wp.kernel
def _curl_kernel(
    fa: wp.array(dtype=wp.vec3),
    fb: wp.array(dtype=wp.vec3),
    ta: wp.array(dtype=wp.vec3),
    tb: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    to_tel: float,
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

    pa = wp.lerp(fa[pr], ta[pr], to_tel)
    pb = wp.lerp(fb[pr], tb[pr], to_tel)

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

    base = 0.055                                        # fat splats: the fibre packs solid/opaque, no see-through holes
    if s >= 2:
        base = 0.050
    rpx = zoom * base / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 11.0))

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
    fog = wp.clamp((depthq - 0.28) * 1.7, 0.0, 0.9)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    u = min(max((time - 0.6) / 4.0, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _camera(time: float):
    # fixed course, no spin: a close-up on the near end's t-loop, hanging out of the corner of the fibre
    # forest, dollying in as the terminal DNA curls into its cap; the fibre trails away behind it.
    u = min(max((time - 0.3) / 4.2, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 20.0 * (1.0 - u) + 13.0 * u
    target = np.array([_LOOPC[0], _LOOPC[1], _LOOPC[2]], np.float32)
    # view the lasso FACE-ON: sit off to the side of the loop plane (a little above and a little outward),
    # so the t-loop reads as a loop and the fibre forest sits behind it
    direction = (_SIDE + 0.30 * _OUT + np.array([0.0, 0.32, 0.0], np.float32)).astype(np.float32)
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww, dist


def _render(width, height, time, mouse, device, cam=None):
    _ensure(device)
    W, H = int(width), int(height)
    to_tel = _schedule(float(time))
    if cam is None:
        ro, uu, vv, ww, dist = _camera(float(time))
    else:
        ro, uu, vv, ww, dist = cam
    dnear = float(dist) - 8.0
    dfar = float(dist) + 60.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _curl_kernel,
        dim=_M,
        inputs=[_fa, _fb, _ta, _tb, _a_col, _b_col, zbuf, elemcol, W, H,
                float(to_tel), *cam, 1.7, dnear, dfar],
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
    name="warp_telomere",
    description=(
        "Process 6 — telomeres. The DNA strand from Process 5 has two ends, so two telomeres: each terminal "
        "stretch (telomere-green) leaves the fibre and curls into a t-loop lasso, the free 3' tip tucking "
        "back to cap and protect the end. Conserving: chained from Process 5's actual fibre, only the ends "
        "reshaped, nothing spawned, fixed camera featuring the cap."
    ),
    renderer=_render,
)
