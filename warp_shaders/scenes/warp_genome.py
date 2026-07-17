"""The genome journey — all six conserving processes chained into one continuous morph.

One point per base pair, carried through the whole ladder the genome library builds one process at a
time: **tokens (the card) -> base pairs -> double helix -> nucleosomes -> 30nm fibre -> chromosome**.
Each stage's shape comes from its own library function (recentred + per-axis normalised to a common box);
the scene morphs every point continuously from one stage to the next while the colour warms from the
token palette to chromosome blue and the camera orbits.

Matter is conserved throughout — the same 182872 points, reconfiguring, never spawned. This is the whole
arc in one scene: the real graphics card carried all the way to a chromosome.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import genome_journey
from ..scene import Scene

_KF, _COLS, _BLUE = genome_journey(sub=2, block=5)
_K, _N, _ = _KF.shape
_KF_FLAT = _KF.reshape(_K * _N, 3)

_kf = _cols = None


def _ensure(device):
    global _kf, _cols
    if _kf is None:
        _kf = wp.array(_KF_FLAT, dtype=wp.vec3, device=device)
        _cols = wp.array(_COLS, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(19)
_IDX_MASK = wp.constant(0x7FFFF)


@wp.kernel
def _splat_kernel(
    kf: wp.array(dtype=wp.vec3),
    cols: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    n: int,
    seg: int,
    frac: float,
    cfrac: float,
    blue: wp.vec3,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    t = wp.tid()
    a = kf[seg * n + t]
    b = kf[(seg + 1) * n + t]
    p = wp.lerp(a, b, frac)                             # morph this stage -> the next
    elemcol[t] = wp.lerp(cols[t], blue, cfrac)          # token palette -> chromosome blue

    rel = p - ro
    cz = wp.dot(rel, ww)
    if cz < 0.05:
        return
    cx = wp.dot(rel, uu)
    cy = wp.dot(rel, vv)
    pfx = zoom * cx / cz * float(height) + 0.5 * float(width) - 0.5
    pfy = 0.5 * float(height) - 0.5 - zoom * cy / cz * float(height)
    px = int(wp.round(pfx))
    py = int(wp.round(pfy))

    rpx = zoom * 0.030 / cz * float(height)
    rad = int(wp.clamp(rpx, 1.0, 5.0))

    depthq = int(wp.clamp((cz - 3.0) / 12.0 * 1022.0, 0.0, 1022.0))
    key = (depthq << _IDX_BITS) | t

    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if float(dx * dx + dy * dy) <= float(rad * rad) + 0.5:
                xx = px + dx
                yy = py + dy
                if xx >= 0 and xx < width and yy >= 0 and yy < height:
                    wp.atomic_min(zbuf, yy, xx, key)


@wp.kernel
def _resolve_kernel(
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
):
    i, j = wp.tid()
    yy = float(i) / float(height)
    bg = wp.vec3(0.015, 0.018, 0.028) * (1.0 - 0.42 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.28 - 0.58 * depthq
    fog = wp.clamp((depthq - 0.55) * 1.7, 0.0, 0.8)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _progress(time: float):
    """Journey parameter u in [0, K-1]; each stage holds ~0.4s at its shape before morphing on."""
    span = float(_K - 1)
    u = min(max(time / 1.25, 0.0), span)                # ~1.9s per stage transition
    seg = min(int(u), _K - 2)
    f = min(max(u - float(seg), 0.0), 1.0)
    f = f * f * (3.0 - 2.0 * f)                         # smoothstep — ease each transition
    cfrac = min(max(u / span, 0.0), 1.0)
    return seg, f, cfrac


def _camera(time: float):
    target = np.array([0.0, 0.0, 0.0], np.float32)
    # orbit through the 3-D stages, then ease to face-on as it lands on the planar chromosome X
    u = min(max(time / 1.25, 0.0), float(_K - 1))
    g = min(max((u - 4.0), 0.0), 1.0)
    g = g * g * (3.0 - 2.0 * g)
    az = (0.4 + 0.28 * time) * (1.0 - g)               # continuous orbit -> settle face-on
    el = 0.16 - 0.06 * g
    dist = 8.2
    ro = target + dist * np.array(
        [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], np.float32
    )
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww


def _render(width, height, time, mouse, device):
    _ensure(device)
    W, H = int(width), int(height)
    seg, frac, cfrac = _progress(float(time))
    ro, uu, vv, ww = _camera(float(time))

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_N, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _splat_kernel,
        dim=_N,
        inputs=[_kf, _cols, zbuf, elemcol, W, H, _N, int(seg), float(frac), float(cfrac),
                wp.vec3(float(_BLUE[0]), float(_BLUE[1]), float(_BLUE[2])), *cam, 1.7],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.38, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_genome",
    description=(
        "The genome journey — all six conserving processes in one morph: the RTX board's 182872 base "
        "pairs carried tokens -> base pairs -> double helix -> nucleosomes -> 30nm fibre -> chromosome, "
        "colour warming to chromosome blue. Conserving throughout: the same points, never spawned."
    ),
    renderer=_render,
)
