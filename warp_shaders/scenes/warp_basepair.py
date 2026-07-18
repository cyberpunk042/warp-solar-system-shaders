"""Process 2 scene — the floating tokens bind into base pairs.

Takes the token cloud from Process 1 (the tokenised card, floating) and binds the tokens **in twos**:
365744 tokens become 182872 base pairs. Over ``time`` each pair's two tokens draw together into a
short coloured rung (A-T / G-C) and the pairs settle into an ordered field — an unwound ladder of base
pairs, order emerging from the token cloud.

Conserving and physical: every token joins exactly one pair (nothing spawned, nothing dropped); the
tokens move continuously from their floating positions to their binding sites. The camera orbits
(never static). This process stops at the base-pair field.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import bind_pairs
from ..scene import Scene

# bind the real card's tokens into base pairs once (static endpoints); per-frame only the binding moves
_BP = bind_pairs(sub=2, block=5)
_P = _BP.n_pairs
_SAMPLES = 5                      # per pair: token A (top), token B (bottom), 3 rung points -> solid bar
_M = _P * _SAMPLES
_HL = 0.13                        # rung half-length at full binding

# ordered binding sites — the pairs settle onto a wide, shallow lattice (the unwound ladder), so the
# base-pair rungs are resolvable instead of stacking into an opaque cloud. Conserving: the tokens just
# drift, continuously, from their floating positions to these sites.
_NX = 432
_SX = 0.115
_SZ = 0.150
_Y = 1.15


def _grid_sites(n):
    gi = np.arange(n)
    gx = gi % _NX
    gz = gi // _NX
    nz = int(np.ceil(n / _NX))
    frac = lambda v: v - np.floor(v)
    jx = (frac(np.sin(gi * 12.9898) * 43758.5453) - 0.5) * (0.7 * _SX)   # break the perfect lattice
    jz = (frac(np.sin(gi * 78.2330 + 2.0) * 43758.5453) - 0.5) * (0.7 * _SZ)  # (kills moire, organic)
    jy = (frac(np.sin(gi * 37.7190 + 4.0) * 43758.5453) - 0.5) * 0.10
    x = (gx - _NX * 0.5) * _SX + jx
    z = (gz - nz * 0.5) * _SZ + jz
    y = _Y + jy
    return np.stack([x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)], axis=1)


_MID_GRID_NP = _grid_sites(_P)

_a_pos = _b_pos = _mid_cloud = _mid_grid = _a_col = _b_col = None


def _ensure(device):
    global _a_pos, _b_pos, _mid_cloud, _mid_grid, _a_col, _b_col
    if _a_pos is None:
        _a_pos = wp.array(_BP.a_pos, dtype=wp.vec3, device=device)
        _b_pos = wp.array(_BP.b_pos, dtype=wp.vec3, device=device)
        _mid_cloud = wp.array(_BP.mid, dtype=wp.vec3, device=device)
        _mid_grid = wp.array(_MID_GRID_NP, dtype=wp.vec3, device=device)
        _a_col = wp.array(_BP.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_BP.b_col, dtype=wp.vec3, device=device)


# packed z-buffer key = (depthQ << 20) | elementIndex ; depthQ in [0,1023], index < 2^20 (1048576)
_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)


@wp.kernel
def _pair_splat_kernel(
    a_pos: wp.array(dtype=wp.vec3),
    b_pos: wp.array(dtype=wp.vec3),
    mid_cloud: wp.array(dtype=wp.vec3),
    mid_grid: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    bind: float,
    hl: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    e = wp.tid()
    pr = e / 5
    s = e - pr * 5

    up = wp.vec3(0.0, 1.0, 0.0)                       # bound pairs align — order emerges from the cloud
    mid_now = wp.lerp(mid_cloud[pr], mid_grid[pr], bind)
    pa = wp.lerp(a_pos[pr], mid_now - up * hl, bind)  # token A: floating -> binding site (top)
    pb = wp.lerp(b_pos[pr], mid_now + up * hl, bind)  # token B: floating -> binding site (bottom)
    ca = a_col[pr]
    cb = b_col[pr]

    p = pa
    col = ca
    if s == 1:
        p = pb
        col = cb
    elif s == 2:
        p = wp.lerp(pa, pb, 0.25)
        col = wp.lerp(ca, cb, 0.25) * bind            # rung fades in only as the pair binds
    elif s == 3:
        p = wp.lerp(pa, pb, 0.5)
        col = wp.lerp(ca, cb, 0.5) * bind
    elif s == 4:
        p = wp.lerp(pa, pb, 0.75)
        col = wp.lerp(ca, cb, 0.75) * bind
    elemcol[e] = col

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

    base = 0.024
    if s >= 2:
        base = 0.017                                  # rung dots a touch smaller than the token dots
    rpx = zoom * base / cz * float(height)
    rad = int(wp.clamp(rpx, 1.0, 3.0))

    depthq = int(wp.clamp((cz - 2.0) / 12.0 * 1023.0, 0.0, 1023.0))
    key = (depthq << _IDX_BITS) | e

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
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1023.0
    shade = 1.2 - 0.55 * depthq
    fog = wp.clamp((depthq - 0.35) * 1.7, 0.0, 0.82)     # far base pairs fade into the dark
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _bind(time: float) -> float:
    """0 at t=0 (the floating token cloud) -> 1 by ~3.4s (all bound + ordered), then holds."""
    u = min(max(time / 3.4, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(u * math.pi)


def _camera(time: float):
    target = np.array([0.0, 1.15, 0.0], np.float32)
    az = 0.5 + 0.2 * time            # gentle orbit — never static
    el = 0.34                        # 3/4 view — the vertical rungs read as an unwound ladder
    dist = 6.0
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
    ro, uu, vv, ww = _camera(float(time))
    bind = _bind(float(time))

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _pair_splat_kernel,
        dim=_M,
        inputs=[_a_pos, _b_pos, _mid_cloud, _mid_grid, _a_col, _b_col, zbuf, elemcol,
                W, H, float(bind), _HL, *cam, 1.7],
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
    name="warp_basepair",
    description=(
        "Process 2 — base-pair bounding. The card's ~366k floating tokens bind in twos into 182872 "
        "base pairs (A-T / G-C coloured rungs) that settle into an ordered field — an unwound ladder. "
        "Conserving: every token joins exactly one pair, nothing spawned; tokens drift continuously."
    ),
    renderer=_render,
)
