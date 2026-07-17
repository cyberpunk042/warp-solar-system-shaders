"""Process 5 scene — the beads-on-a-string coil into the 30nm fibre.

Takes the nucleosome beads-on-a-string from Process 4 and does the next level of packing: the string
coils into a solenoid — ~6 nucleosomes per turn wound around a common axis into the thick 30nm chromatin
fibre. Over ``time`` the string reels onto the solenoid while the camera travels along the fibre.

Conserving and physical: each bead is carried whole from the string onto the solenoid (a rigid
translation; its inner wrap untouched). Every base pair placed exactly once. This process stops at the
30nm fibre.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import coil_fiber
from ..scene import Scene

_FB = coil_fiber(sub=2, block=5)
_P = _FB.n_pairs

_string = _fiber = _cols = None


def _ensure(device):
    global _string, _fiber, _cols
    if _string is None:
        _string = wp.array(_FB.string, dtype=wp.vec3, device=device)
        _fiber = wp.array(_FB.fiber, dtype=wp.vec3, device=device)
        _cols = wp.array(_FB.colors, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(19)
_IDX_MASK = wp.constant(0x7FFFF)


@wp.kernel
def _splat_kernel(
    string: wp.array(dtype=wp.vec3),
    fiber: wp.array(dtype=wp.vec3),
    cols: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    coil: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    t = wp.tid()
    p = wp.lerp(string[t], fiber[t], coil)             # beads-on-a-string -> solenoid fibre
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

    rpx = zoom * 0.058 / cz * float(height)
    rad = int(wp.clamp(rpx, 1.0, 6.0))

    depthq = int(wp.clamp((cz - 3.0) / 24.0 * 1022.0, 0.0, 1022.0))
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
    bg = wp.vec3(0.016, 0.020, 0.030) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1022.0
    shade = 1.28 - 0.6 * depthq
    fog = wp.clamp((depthq - 0.5) * 1.7, 0.0, 0.82)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _coil(time: float) -> float:
    """0 (beads on a string) -> 1 (30nm solenoid) by ~3.2s, then holds."""
    u = min(max(time / 3.2, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(u * math.pi)


def _camera(time: float, cx: float):
    target = np.array([cx, 0.0, 0.0], np.float32)
    az = 0.18 * math.sin(time * 0.3)                  # gentle sway — never static
    el = 0.16
    dist = 18.0
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
    coil = _coil(float(time))
    cx = -8.0 + 3.0 * float(time)                     # travel along the fibre
    ro, uu, vv, ww = _camera(float(time), cx)

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_P, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _splat_kernel,
        dim=_P,
        inputs=[_string, _fiber, _cols, zbuf, elemcol, W, H, float(coil), *cam, 1.7],
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
    name="warp_fiber",
    description=(
        "Process 5 — the 30nm fibre. The nucleosome beads-on-a-string coil into a solenoid (~6 "
        "nucleosomes per turn, 915 beads). Conserving: each bead carried whole onto the fibre, nothing "
        "spawned; the string reels onto the solenoid continuously."
    ),
    renderer=_render,
)
