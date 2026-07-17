"""Process 3 scene — the base pairs wind into the DNA double helix (the whole strand, whole process).

Chains directly from Process 2: it starts from the exact base-pair field Process 2 produced (every
pair's two tokens on a rung) and physically winds it. First the field of rungs gathers into one straight
ladder, in sequence; then the ladder **twists** about its axis into the right-handed double helix — the
two tokens of each rung tracing the two backbones.

Conserving and physical: no point is created or destroyed and none teleports — each token moves
continuously from where Process 2 left it, through the flat ladder, onto the helix. The camera is fixed;
the whole strand (all 182872 base pairs) is in frame the whole time, so the entire process is visible.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..engine import post
from ..genome import wind_helix
from ..scene import Scene

_HX = wind_helix(sub=2, block=5)
_P = _HX.n_pairs
_SAMPLES = 4                      # per pair: backbone token A, backbone token B, 2 rung points
_M = _P * _SAMPLES

_field_a = _field_b = _a_col = _b_col = None


def _ensure(device):
    global _field_a, _field_b, _a_col, _b_col
    if _field_a is None:
        _field_a = wp.array(_HX.field_a, dtype=wp.vec3, device=device)
        _field_b = wp.array(_HX.field_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_HX.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_HX.b_col, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)


@wp.func
def _helix_end(pr: int, p: int, off: float, gather: float, twist: float,
               radius: float, height: float, dtheta: float) -> wp.vec3:
    # rung centre on the ladder axis (pairs stacked in sequence along y)
    y = (float(pr) / float(p) - 0.5) * height
    theta = twist * float(pr) * dtheta + off           # twist=0 -> flat ladder; twist=1 -> full helix
    return wp.vec3(radius * gather * wp.cos(theta), y, radius * gather * wp.sin(theta))


@wp.kernel
def _wind_kernel(
    field_a: wp.array(dtype=wp.vec3),
    field_b: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    p: int,
    to_ladder: float,       # 0 = base-pair field (Process 2 output), 1 = gathered into the ladder
    twist: float,           # 0 = flat ladder, 1 = full double helix
    radius: float,
    hheight: float,
    dtheta: float,
    groove: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    e = wp.tid()
    pr = e / 4
    s = e - pr * 4

    # ladder/helix endpoints for this pair's two tokens (gather scales the rung out from the axis)
    ha = _helix_end(pr, p, 0.0, 1.0, twist, radius, hheight, dtheta)
    hb = _helix_end(pr, p, groove, 1.0, twist, radius, hheight, dtheta)
    # continuous motion: Process-2 field -> ladder/helix endpoint
    pa = wp.lerp(field_a[pr], ha, to_ladder)
    pb = wp.lerp(field_b[pr], hb, to_ladder)

    if s == 0:
        pos = pa
        col = a_col[pr]
    elif s == 1:
        pos = pb
        col = b_col[pr]
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.34)
        col = wp.lerp(a_col[pr], b_col[pr], 0.34)
    else:
        pos = wp.lerp(pa, pb, 0.66)
        col = wp.lerp(a_col[pr], b_col[pr], 0.66)
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

    base = 0.020
    if s >= 2:
        base = 0.014
    rpx = zoom * base / cz * float(height_px)
    rad = int(wp.clamp(rpx, 1.0, 4.0))

    depthq = int(wp.clamp((cz - 4.0) / 12.0 * 1022.0, 0.0, 1022.0))
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
    shade = 1.22 - 0.5 * depthq
    img[i, j] = elemcol[idx] * shade


def _schedule(time: float):
    """Whole process on one timeline: hold the field, gather into the ladder, then twist into the helix."""
    def ss(a, b):
        u = min(max((time - a) / (b - a), 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)
    to_ladder = ss(0.4, 2.4)      # base-pair field -> straight ladder
    twist = ss(2.7, 5.7)          # flat ladder -> full double helix
    return to_ladder, twist


def _camera(time: float):
    # no rotation — only a straight push-in (dolly) from the wide base-pair field to the tight helix,
    # tracking the gather. The winding itself is the motion.
    u = min(max((time - 0.2) / 3.3, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 15.5 * (1.0 - u) + 9.2 * u
    cy = 1.15 * (1.0 - u) + 0.3 * u
    ro = np.array([0.0, cy, dist], np.float32)
    target = np.array([0.0, cy * 0.5, 0.0], np.float32)
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww


def _render(width, height, time, mouse, device):
    _ensure(device)
    W, H = int(width), int(height)
    to_ladder, twist = _schedule(float(time))
    ro, uu, vv, ww = _camera(float(time))

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _wind_kernel,
        dim=_M,
        inputs=[_field_a, _field_b, _a_col, _b_col, zbuf, elemcol, W, H, _P,
                float(to_ladder), float(twist), _HX.radius, _HX.height, _HX.dtheta, _HX.groove,
                *cam, 1.7],
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
    name="warp_helix",
    description=(
        "Process 3 — the double helix. The base-pair field from Process 2 gathers into a straight ladder "
        "and twists into the right-handed DNA double helix — the two tokens of each rung tracing the two "
        "backbones. Conserving: chained from Process 2's actual output, nothing spawned, fixed camera, "
        "the whole strand winding in frame."
    ),
    renderer=_render,
)
