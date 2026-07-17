"""Process 5 scene — the beads-on-a-string coil into 30 nm solenoid fibres.

Chains directly from Process 4: it starts from the exact nucleosome beads Process 4 produced and coils
them. The "beads on a string" wind into a **~30 nm fibre** at ~6 nucleosomes per turn — and because a
whole row of ~36 beads coils into one fibre, the 1663 beads **funnel into ~47 fibres** (the first real
drop in count) while compacting ~6× along the fibre axis.

Conserving and physical: each nucleosome bead moves as a **rigid unit** — its wrapped ring of DNA is
carried along onto the solenoid — so every base pair is reused, nothing spawned, nothing teleports. The
camera holds a fixed course (a slow dolly, no spin) as the wide carpet of beads gathers and coils into the
field of fibres — the whole field, the whole coil, in frame.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..genome import coil_fibre
from ..scene import Scene

_FB = coil_fibre(sub=2, block=5)
_P = _FB.n_pairs
_SAMPLES = 4
_M = _P * _SAMPLES

_ba = _bb = _fa = _fb = _a_col = _b_col = None


def _ensure(device):
    global _ba, _bb, _fa, _fb, _a_col, _b_col
    if _ba is None:
        _ba = wp.array(_FB.bead_a, dtype=wp.vec3, device=device)
        _bb = wp.array(_FB.bead_b, dtype=wp.vec3, device=device)
        _fa = wp.array(_FB.fib_a, dtype=wp.vec3, device=device)
        _fb = wp.array(_FB.fib_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_FB.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_FB.b_col, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.46, 0.53, 0.66))


@wp.kernel
def _coil_kernel(
    ba: wp.array(dtype=wp.vec3),
    bb: wp.array(dtype=wp.vec3),
    fa: wp.array(dtype=wp.vec3),
    fb: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    to_fibre: float,        # 0 = Process-4 beads on a string, 1 = coiled into 30 nm fibres
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

    pa = wp.lerp(ba[pr], fa[pr], to_fibre)
    pb = wp.lerp(bb[pr], fb[pr], to_fibre)

    if s == 0:
        pos = pa
        col = _BACKBONE
    elif s == 1:
        pos = pb
        col = _BACKBONE
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
        col = a_col[pr]
    else:
        pos = wp.lerp(pa, pb, 0.64)
        col = b_col[pr]
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
    rad = int(wp.clamp(rpx, 1.0, 6.0))

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
    shade = 1.28 - 1.12 * depthq
    fog = wp.clamp((depthq - 0.20) * 2.3, 0.0, 0.94)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    u = min(max((time - 0.5) / 4.0, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def _camera(time: float):
    # fixed course, no spin: as the wide bead carpet gathers and coils into the narrow band of fibres,
    # the camera dollies in and looks along the band so the solenoid coils read.
    u = min(max((time - 0.3) / 4.2, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 46.0 * (1.0 - u) + 33.0 * u
    target = np.array([0.0, 0.0, 0.0], np.float32)
    direction = np.array([0.05, 0.30 + 0.26 * u, 1.0], np.float32)   # tilt down onto the coiled stack
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
    to_fibre = _schedule(float(time))
    ro, uu, vv, ww, dist = _camera(float(time))
    dnear = 1.5
    dfar = float(dist) + 40.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _coil_kernel,
        dim=_M,
        inputs=[_ba, _bb, _fa, _fb, _a_col, _b_col, zbuf, elemcol, W, H,
                float(to_fibre), *cam, 1.7, dnear, dfar],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_fibre",
    description=(
        "Process 5 — the 30 nm fibre. The nucleosome beads from Process 4 coil into solenoid fibres at ~6 "
        "beads per turn; a row of beads per fibre, so the 1663 beads funnel into ~47 fibres and compact ~6×. "
        "Conserving: chained from Process 4's actual beads, each bead rigid-moved onto the solenoid, nothing "
        "spawned, fixed camera, the whole field coiling in frame."
    ),
    renderer=_render,
)
