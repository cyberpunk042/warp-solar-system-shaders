"""Process 3 scene — the base pairs wind into the DNA double helix.

Takes the base-pair field from Process 2 and winds it into DNA: each base pair is a rung, and the pair's
two tokens are the two points on the two backbones that spiral around the axis (~10.5 base pairs per
turn). Over ``time`` the strand winds up from a loose ladder into the tight right-handed double helix,
and the camera climbs the molecule while it turns.

Conserving and physical: every base pair is placed exactly once (nothing spawned); the backbones are the
pairs' own tokens in sequence. The strand is very long — the camera frames a clear section of it (that
length is why the next steps coil it into a chromosome). This process stops at the double helix.
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
_SAMPLES = 10                     # per pair: 3 backbone-1 beads, 3 backbone-2 beads, 4 rung points
_M = _P * _SAMPLES

_RADIUS = 0.72
_RISE = 0.095
_DTHETA = 2.0 * math.pi / 10.5    # ~10.5 base pairs per turn
_GROOVE = 2.7                     # strands offset — rungs read as clear cross-bars, grooves show

_rung_a = _rung_b = None


def _ensure(device):
    global _rung_a, _rung_b
    if _rung_a is None:
        _rung_a = wp.array(_HX.rung_a, dtype=wp.vec3, device=device)
        _rung_b = wp.array(_HX.rung_b, dtype=wp.vec3, device=device)


_INIT = wp.constant(0x7FFFFFFF)
_IDX_BITS = wp.constant(21)
_IDX_MASK = wp.constant(0x1FFFFF)
_BACKBONE = wp.constant(wp.vec3(0.72, 0.80, 0.95))


@wp.func
def _strand(pr: float, radius: float, rise: float, dtheta: float, off: float,
            wind: float, y0: float) -> wp.vec3:
    theta = pr * dtheta * wind + off
    return wp.vec3(radius * wp.cos(theta), pr * rise - y0, radius * wp.sin(theta))


@wp.kernel
def _helix_splat_kernel(
    rung_a: wp.array(dtype=wp.vec3),
    rung_b: wp.array(dtype=wp.vec3),
    zbuf: wp.array2d(dtype=wp.int32),
    elemcol: wp.array(dtype=wp.vec3),
    width: int,
    height: int,
    wind: float,
    radius: float,
    rise: float,
    dtheta: float,
    groove: float,
    y0: float,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    zoom: float,
):
    e = wp.tid()
    pr = e / 10
    s = e - pr * 10
    fp = float(pr)

    # this pair's two backbone points and the next pair's (to fill the backbone continuously)
    s1 = _strand(fp, radius, rise, dtheta, 0.0, wind, y0)
    s2 = _strand(fp, radius, rise, dtheta, groove, wind, y0)
    s1n = _strand(fp + 1.0, radius, rise, dtheta, 0.0, wind, y0)
    s2n = _strand(fp + 1.0, radius, rise, dtheta, groove, wind, y0)

    is_backbone = 1
    if s == 0:
        p = s1
    elif s == 1:
        p = wp.lerp(s1, s1n, 0.333)
    elif s == 2:
        p = wp.lerp(s1, s1n, 0.667)
    elif s == 3:
        p = s2
    elif s == 4:
        p = wp.lerp(s2, s2n, 0.333)
    elif s == 5:
        p = wp.lerp(s2, s2n, 0.667)
    else:
        is_backbone = 0
        f = 0.15 + 0.233 * float(s - 6)               # rung fill: 0.15, 0.383, 0.617, 0.85
        p = wp.lerp(s1, s2, f)

    col = _BACKBONE
    if is_backbone == 0:
        f = 0.15 + 0.233 * float(s - 6)
        col = wp.lerp(rung_a[pr], rung_b[pr], f)       # base colours graded across the rung
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

    base = 0.078                                       # backbone beads overlap into a ribbon
    if is_backbone == 0:
        base = 0.046                                   # rung dots
    rpx = zoom * base / cz * float(height)
    rad = int(wp.clamp(rpx, 1.0, 11.0))

    depthq = int(wp.clamp((cz - 2.0) / 14.0 * 1022.0, 0.0, 1022.0))
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
    bg = wp.vec3(0.016, 0.020, 0.030) * (1.0 - 0.45 * yy)
    key = zbuf[i, j]
    if key == _INIT:
        img[i, j] = bg
        return
    idx = key & _IDX_MASK
    depthq = float((key >> _IDX_BITS) & 0x3FF) / 1023.0
    shade = 1.25 - 0.55 * depthq
    fog = wp.clamp((depthq - 0.4) * 1.6, 0.0, 0.8)
    img[i, j] = wp.lerp(elemcol[idx] * shade, bg, fog)


def _wind(time: float) -> float:
    """0.18 (a loose ladder) -> 1.0 (the tight double helix) by ~3.2s, then holds."""
    u = min(max(time / 3.2, 0.0), 1.0)
    return 0.18 + 0.82 * (0.5 - 0.5 * math.cos(u * math.pi))


def _camera(time: float, y0: float):
    target = np.array([0.0, 0.0, 0.0], np.float32)   # the section is centred at the origin via y0
    az = 0.5 + 0.35 * time                            # orbit — never static
    el = 0.12
    dist = 6.6
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
    wind = _wind(float(time))
    # climb the molecule slowly; the framed section is centred by y0
    center_pair = 900.0 + 26.0 * float(time)
    y0 = center_pair * _RISE
    ro, uu, vv, ww = _camera(float(time), y0)

    zbuf = wp.full((H, W), 0x7FFFFFFF, dtype=wp.int32, device=device)
    elemcol = wp.zeros(_M, dtype=wp.vec3, device=device)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    cam = (wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
           wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]))
    wp.launch(
        _helix_splat_kernel,
        dim=_M,
        inputs=[_rung_a, _rung_b, zbuf, elemcol, W, H, float(wind),
                _RADIUS, _RISE, _DTHETA, _GROOVE, float(y0), *cam, 1.7],
        device=device,
    )
    wp.launch(_resolve_kernel, dim=(H, W), inputs=[zbuf, elemcol, img, W, H], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.0, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.32)
    return ldr


SCENE = Scene(
    name="warp_helix",
    description=(
        "Process 3 — the double helix. The 182872 base pairs wind into DNA: each pair a rung, its two "
        "tokens the two spiralling backbones (~10.5 bp/turn). Conserving: every pair placed once, "
        "nothing spawned; the camera climbs a clear section of the long strand as it winds up."
    ),
    renderer=_render,
)
