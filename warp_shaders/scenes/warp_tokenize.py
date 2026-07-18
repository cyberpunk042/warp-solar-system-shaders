"""Process 1 scene — the graphics card becomes a cloud of tokens floating in the air.

The real RTX board is tokenised by :func:`warp_shaders.genome.tokenize_card` — every occupied bit of
the card is one token, coloured by its merge-codec type id. This scene renders those tokens directly:
a Warp **splat** kernel projects each of the ~366k tokens to the screen and adds a soft glow, so the
whole token stream is on screen at once (raymarching that many discrete tokens is infeasible).

Matter is conserved and physics is not broken: the tokens ARE the card's own bits. Over ``time`` they
lift and spread along smooth continuous trajectories from their home voxels — the card that is no more
a card but a bunch of tokens floating in the air. No token is spawned; the count never changes.

This process STOPS at the token cloud. Base-pair bounding is a separate process.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import tokenize_card
from ..scene import Scene

# tokenise the real board once (static homes + type colours); per-frame only the dispersion moves
_CLOUD = tokenize_card(sub=2, block=5)
_HOMES_NP = _CLOUD.positions
_COLS_NP = _CLOUD.colors
_N = _CLOUD.n

_homes = None
_cols = None


def _ensure_arrays(device):
    global _homes, _cols
    if _homes is None:
        _homes = wp.array(_HOMES_NP, dtype=wp.vec3, device=device)
        _cols = wp.array(_COLS_NP, dtype=wp.vec3, device=device)


@wp.func
def _rand(seed: float) -> float:
    v = wp.sin(seed) * 43758.5453
    return v - wp.floor(v)


# packed z-buffer key = (depthQ << 19) | tokenIndex ; depthQ in [0,2047], index < 2^19 (524288)
_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(19)
_IDX_MASK = wp.constant(0x7FFFF)


@wp.func
def _token_pos(h: wp.vec3, t: int, disperse: float, tsec: float) -> wp.vec3:
    r1 = _rand(float(t) * 12.9898 + 0.5)
    r2 = _rand(float(t) * 78.2330 + 1.3)
    r3 = _rand(float(t) * 37.7190 + 2.7)
    # conserving dispersion: continuous drift from the home voxel — rise + outward spread + float
    ang = r1 * 6.2831853
    outward = wp.vec3(wp.cos(ang), 0.0, wp.sin(ang))
    spread = 0.8 + 1.5 * r2
    rise = 0.4 + 1.7 * r3
    turb = 0.07 * wp.sin(tsec * 0.7 + float(t) * 0.017)
    return h + outward * (disperse * spread + turb) + wp.vec3(0.0, disperse * rise, 0.0)


@wp.kernel
def _zsplat_kernel(
    homes: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    width: int,
    height: int,
    disperse: float,
    tsec: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    t = wp.tid()
    p = _token_pos(homes[t], t, disperse, tsec)

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

    rpx = zoom * 0.024 / cz * float(height)      # token screen radius (perspective)
    rad = int(wp.clamp(rpx, 1.0, 3.0))

    depthq = int(wp.clamp((cz - 2.0) / 12.0 * 2047.0, 0.0, 2047.0))
    key = (depthq << _IDX_BITS) | t              # nearest token (smallest depth) wins the pixel

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
    cols: wp.array(dtype=wp.vec3),
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
):
    i, j = wp.tid()
    # dark graded backdrop
    yy = float(i) / float(height)
    bg = wp.vec3(0.018, 0.022, 0.032) * (1.0 - 0.5 * yy)

    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x7FF)
    shade = 1.18 - 0.5 * (depthq / 2047.0)       # nearer tokens a touch brighter
    img[i, j] = cols[idx] * shade


def _disperse(time: float) -> float:
    """0 at t=0 (tight card) -> 1 by ~3.2s (floating cloud), then holds. Smooth, continuous."""
    u = min(max(time / 3.2, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(u * math.pi)


def _camera(time: float):
    target = np.array([0.0, 0.12, 0.0], np.float32)
    az = 0.62                           # fixed — the card stays put; the tokenisation is the motion
    el = 0.34
    dist = 7.6
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
    _ensure_arrays(device)
    W, H = int(width), int(height)
    ro, uu, vv, ww = _camera(float(time))
    disp = _disperse(float(time))

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _zsplat_kernel,
        dim=_N,
        inputs=[_homes, zbuf, W, H, float(disp), float(time), *cam, 1.7],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, _cols, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_tokenize",
    description=(
        "Process 1 — tokenization. The real RTX board is turned into ~366k tokens (every bit of the "
        "card, coloured by merge-codec type), rising into a cloud of tokens floating in the air. "
        "Conserving transform: the tokens are the card's own matter, nothing spawned."
    ),
    renderer=_render,
)
