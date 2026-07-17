"""Process 6 scene — the 30nm fibre folds into the metaphase chromosome (the blue X).

Takes the 30nm fibre from Process 5 and condenses it: the long fibre gathers and folds into looped
domains that pack the shape of the metaphase chromosome — the blue **X**, two chromatid arms fat and
rounded at the tips, pinched at the centromere with its two lighter nodes. Over ``time`` the gathered
fibre folds into the X while it turns.

Conserving and physical: every base pair (every bit of the original card) is packed into the chromosome
body exactly once — nothing spawned. Each point travels a straight continuous line into place. This is
where the whole ladder lands: the card, become a chromosome.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import fold_chromosome
from ..scene import Scene

_CH = fold_chromosome(sub=2, block=5)
_P = _CH.n_pairs

_fiber = _chromo = _cols = None


def _ensure(device):
    global _fiber, _chromo, _cols
    if _fiber is None:
        _fiber = wp.array(_CH.fiber, dtype=wp.vec3, device=device)
        _chromo = wp.array(_CH.chromo, dtype=wp.vec3, device=device)
        _cols = wp.array(_CH.colors, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(19)
_IDX_MASK = wp.constant(0x7FFFF)
_GATHER = wp.constant(wp.vec3(0.016, 1.0, 1.0))     # squash the long fibre into a gathered bundle


@wp.kernel
def _splat_kernel(
    fiber: wp.array(dtype=wp.vec3),
    chromo: wp.array(dtype=wp.vec3),
    cols: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    fold: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    t = wp.tid()
    f = fiber[t]
    start = wp.vec3(f[0] * _GATHER[0], f[1] * _GATHER[1], f[2] * _GATHER[2])
    p = wp.lerp(start, chromo[t], fold)                # gathered fibre -> folded chromosome X
    elemcol[t] = cols[t]

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

    rpx = zoom * 0.046 / cz * float(height)
    rad = int(wp.clamp(rpx, 1.0, 6.0))

    depthq = int(wp.clamp((cz - 4.0) / 10.0 * 1022.0, 0.0, 1022.0))
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
    bg = wp.vec3(0.015, 0.018, 0.028) * (1.0 - 0.4 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.3 - 0.6 * depthq                          # near side of the arms brighter (roundness)
    img[i, j] = elemcol[idx] * shade


def _fold(time: float) -> float:
    """0.12 (gathered fibre bundle) -> 1 (the folded chromosome X) by ~3.2s, then holds."""
    u = min(max(time / 3.2, 0.0), 1.0)
    return 0.12 + 0.88 * (0.5 - 0.5 * math.cos(u * math.pi))


def _camera(time: float):
    target = np.array([0.0, 0.0, 0.0], np.float32)
    az = 0.30 * math.sin(time * 0.35)                  # near face-on (the X spreads in the xy-plane)
    el = 0.12
    dist = 8.2
    ro = target + dist * np.array(
        [math.cos(el) * math.sin(az), math.sin(el), math.cos(el) * math.cos(az)], np.float32
    )
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    up = np.array([0.0, 1.0, 0.0], np.float32)
    uu = np.cross(ww, up)
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    # gentle in-plane roll — the X turns like a pinwheel while staying face-on (never static)
    roll = 0.16 * time
    cr, sr = math.cos(roll), math.sin(roll)
    uu, vv = cr * uu + sr * vv, -sr * uu + cr * vv
    return ro, uu, vv, ww


def _render(width, height, time, mouse, device):
    _ensure(device)
    W, H = int(width), int(height)
    fold = _fold(float(time))
    ro, uu, vv, ww = _camera(float(time))

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_P, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _splat_kernel,
        dim=_P,
        inputs=[_fiber, _chromo, _cols, zbuf, elemcol, W, H, float(fold), *cam, 1.7],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5, passes=3)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.32)
    return ldr


SCENE = Scene(
    name="warp_chromosome",
    description=(
        "Process 6 — the metaphase chromosome. The 30nm fibre folds and condenses into the blue X: two "
        "chromatid arms, pinched centromere with its nodes, all 182872 base pairs (every bit of the "
        "card) packed in. Conserving: nothing spawned; the whole ladder lands on the chromosome."
    ),
    renderer=_render,
)
