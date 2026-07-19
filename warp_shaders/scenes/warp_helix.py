"""Process 3 scene — the base pairs wind into MANY DNA double helices (the whole field, whole process).

Chains directly from Process 2: it starts from the exact base-pair field Process 2 produced (every
pair's two tokens on a rung) and physically winds it. A double helix only holds ~100 base pairs, so the
182872 pairs are grouped in sequence into 1663 short helices. First each group's rungs gather from the
Process-2 field into its own little straight ladder; then every ladder **twists** about its axis into a
real-proportioned right-handed double helix (10.5 base pairs per turn) — the two tokens of each rung
tracing that helix's two backbones. The whole **field of double helices** is in frame the whole time.

Conserving and physical: no point is created or destroyed and none teleports — each token moves
continuously from where Process 2 left it, into its ladder, onto its helix. The camera holds a fixed
3/4 view and only dollies in (no spin), so the entire process — the gather and the winding — is visible.
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
_G = _HX.bp_per_helix
_SAMPLES = 4                      # per pair: backbone token A, backbone token B, 2 rung points
_M = _P * _SAMPLES

_field_a = _field_b = _a_col = _b_col = _centers = None


def _ensure(device):
    global _field_a, _field_b, _a_col, _b_col, _centers
    if _field_a is None:
        _field_a = wp.array(_HX.field_a, dtype=wp.vec3, device=device)
        _field_b = wp.array(_HX.field_b, dtype=wp.vec3, device=device)
        _a_col = wp.array(_HX.a_col, dtype=wp.vec3, device=device)
        _b_col = wp.array(_HX.b_col, dtype=wp.vec3, device=device)
        _centers = wp.array(_HX.centers, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(20)
_IDX_MASK = wp.constant(0xFFFFF)
_BACKBONE = wp.constant(wp.vec3(0.46, 0.53, 0.66))   # sugar-phosphate backbones — muted so they don't blow out


@wp.func
def _helix_end(c: wp.vec3, l: int, g: int, off: float, twist: float,
               radius: float, height: float, dtheta: float) -> wp.vec3:
    # rung centre stacked in sequence along this helix's own vertical axis, centred on c
    y = (float(l) / float(g) - 0.5) * height
    theta = twist * float(l) * dtheta + off            # twist=0 -> flat ladder; twist=1 -> full helix
    return wp.vec3(c[0] + radius * wp.cos(theta), c[1] + y, c[2] + radius * wp.sin(theta))


@wp.kernel
def _wind_kernel(
    field_a: wp.array(dtype=wp.vec3),
    field_b: wp.array(dtype=wp.vec3),
    a_col: wp.array(dtype=wp.vec3),
    b_col: wp.array(dtype=wp.vec3),
    centers: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height_px: int,
    g: int,
    to_ladder: float,       # 0 = base-pair field (Process 2 output), 1 = gathered into its ladder
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
    dnear: float,
    dfar: float,
):
    e = wp.tid()
    pr = e / 4
    s = e - pr * 4
    hg = pr / g                         # which helix this pair belongs to
    l = pr - hg * g                     # its rung index within that helix
    c = centers[hg]

    # ladder/helix endpoints for this pair's two tokens (about this helix's own centre)
    ha = _helix_end(c, l, g, 0.0, twist, radius, hheight, dtheta)
    hb = _helix_end(c, l, g, groove, twist, radius, hheight, dtheta)
    # continuous motion: Process-2 field -> this helix's ladder/helix endpoint
    pa = wp.lerp(field_a[pr], ha, to_ladder)
    pb = wp.lerp(field_b[pr], hb, to_ladder)

    if s == 0:
        pos = pa
        col = _BACKBONE                                # the two backbones read as smooth pale ribbons
    elif s == 1:
        pos = pb
        col = _BACKBONE
    elif s == 2:
        pos = wp.lerp(pa, pb, 0.36)
        col = a_col[pr]                                # base-pair rungs keep their A/T/G/C colour
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

    base = 0.030                                       # backbone beads overlap into a smooth ribbon
    if s >= 2:
        base = 0.016                                   # rungs a touch fatter, so their A/T/G/C colour reads
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
    shade = 1.28 - 1.12 * depthq                        # near helices bright, far ones dark -> depth
    fog = wp.clamp((depthq - 0.20) * 2.3, 0.0, 0.94)    # the far forest recedes steeply into the dark
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _schedule(time: float):
    """Whole process on one timeline: hold the field, gather into ladders, then twist into helices."""
    def ss(a, b):
        u = min(max((time - a) / (b - a), 0.0), 1.0)
        return u * u * (3.0 - 2.0 * u)
    to_ladder = ss(0.4, 2.6)      # base-pair field -> straight ladders
    twist = ss(2.9, 5.8)          # flat ladders -> full double helices
    return to_ladder, twist


def _camera(time: float):
    # fixed 3/4 view (a constant look direction — no spin); only a straight dolly in, from the wide
    # base-pair field to the tight forest of helices. The winding itself is the motion.
    u = min(max((time - 0.2) / 3.4, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 52.0 * (1.0 - u) + 44.0 * u
    ty = 0.6 * (1.0 - u) + 0.0 * u
    target = np.array([0.0, ty, 0.0], np.float32)
    direction = np.array([0.10, 0.30, 1.0], np.float32)   # outside the front edge, tilted down: whole field a receding wedge
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
    to_ladder, twist = _schedule(float(time))
    if cam is None:
        ro, uu, vv, ww, dist = _camera(float(time))
    else:
        ro, uu, vv, ww, dist = cam
    dnear = 1.5
    dfar = float(dist) + 34.0

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _wind_kernel,
        dim=_M,
        inputs=[_field_a, _field_b, _a_col, _b_col, _centers, zbuf, elemcol, W, H, _G,
                float(to_ladder), float(twist), _HX.radius, _HX.height, _HX.dtheta, _HX.groove,
                *cam, 1.7, dnear, dfar],
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
        "Process 3 — the double helices. The base-pair field from Process 2 gathers, in groups of ~110, "
        "into straight ladders that twist into real-proportioned right-handed DNA double helices — the "
        "two tokens of each rung tracing the two backbones. A double helix holds only ~100 base pairs, so "
        "the 182872 pairs make a whole field of 1663 helices. Conserving: chained from Process 2's actual "
        "output, nothing spawned, fixed camera, the whole field winding in frame."
    ),
    renderer=_render,
)
