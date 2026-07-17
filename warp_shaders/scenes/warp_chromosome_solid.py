"""The chromosome as a solid, lit object — a ray-marched SDF, not a point cloud.

The genome ladder (Processes 1-8) shows the DNA condensing as its real, conserved base pairs. This scene is
the **payoff render**: the finished chromosome as the shape everyone pictures — the iconic **metaphase X** —
rendered as a signed-distance field, sphere-traced with real lighting (key light, soft self-shadows, ambient
occlusion, PBR), so it reads like a stained chromosome under a microscope: solid, opaque, banded.

Shape (physical): a chromatid is a smooth rod whose radius follows the real silhouette — a fat **arm**,
tapering to a **rounded telomere cap**, pinched to a sharp **centromere** waist at the middle. Two identical
**sister chromatids** join at the centromere and cross into the **X**, four arms in all, carrying faint
**G-bands** across their length. Over time the whole thing tells the story: a long thin wavy chromatin thread
**condenses** (coils and thickens) into one chromatid, then the sister **splays** open into the X — a slow
3/4 course, no spin.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..procedural.sdf import op_smooth_union
from ..scene import Scene

_STEPS = wp.constant(180)
_MAXD = wp.constant(80.0)


@wp.func
def _rprof(yn: float) -> float:
    # smooth radius profile along a chromatid arm: fat mid-arm, rounded toward the tip, a sharp centromere
    # constriction at the middle.
    a = wp.abs(yn)
    d = (a - 0.55) / 0.32
    prof = 0.34 + 0.68 * wp.exp(-d * d)
    cc = yn / 0.11
    prof = prof * (1.0 - 0.42 * wp.exp(-cc * cc))                 # the centromere primary constriction
    return prof


@wp.func
def _rotz(p: wp.vec3, ang: float) -> wp.vec3:
    c = wp.cos(ang)
    s = wp.sin(ang)
    return wp.vec3(p[0] * c - p[1] * s, p[0] * s + p[1] * c, p[2])


@wp.func
def _rod(p: wp.vec3, hh: float, rmax: float, wob: float) -> float:
    y = p[1]
    ax = wob * wp.sin(y * 2.3 + 0.6)                              # wavy axis when loose, straight when condensed
    az = wob * wp.cos(y * 1.9)
    cy = wp.clamp(y, -hh, hh)                                     # clamp to the segment → rounded caps
    px = p[0] - ax
    pz = p[2] - az
    rad = rmax * _rprof(cy / hh) * (1.0 + 0.020 * wp.sin(cy * 9.0))   # faint surface banding
    return wp.sqrt(px * px + (y - cy) * (y - cy) + pz * pz) - rad


@wp.func
def _map(p: wp.vec3, hh: float, rmax: float, tilt: float, wob: float) -> float:
    a = _rod(_rotz(p, -tilt), hh, rmax, wob)                      # sister A (object tilted +tilt)
    b = _rod(_rotz(p, tilt), hh, rmax, wob)                       # sister B (object tilted -tilt)
    return op_smooth_union(a, b, 0.30)                            # joined at the centromere


@wp.func
def _normal(p: wp.vec3, hh: float, rmax: float, tilt: float, wob: float) -> wp.vec3:
    ex = wp.vec3(0.004, 0.0, 0.0)
    ey = wp.vec3(0.0, 0.004, 0.0)
    ez = wp.vec3(0.0, 0.0, 0.004)
    dx = _map(p + ex, hh, rmax, tilt, wob) - _map(p - ex, hh, rmax, tilt, wob)
    dy = _map(p + ey, hh, rmax, tilt, wob) - _map(p - ey, hh, rmax, tilt, wob)
    dz = _map(p + ez, hh, rmax, tilt, wob) - _map(p - ez, hh, rmax, tilt, wob)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _soft_shadow(p: wp.vec3, ld: wp.vec3, hh: float, rmax: float, tilt: float, wob: float) -> float:
    res = float(1.0)
    t = float(0.03)
    for _ in range(38):
        h = _map(p + ld * t, hh, rmax, tilt, wob)
        if h < 0.001:
            return 0.0
        res = wp.min(res, 12.0 * h / t)
        t += wp.clamp(h, 0.015, 0.25)
        if t > 14.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, hh: float, rmax: float, tilt: float, wob: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.02 + 0.11 * float(k)
        dd = _map(p + n * hr, hh, rmax, tilt, wob)
        occ += (hr - dd) * sca
        sca *= 0.82
    return wp.clamp(1.0 - 2.2 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height_px: int,
    ro: wp.vec3,
    uu: wp.vec3,
    vv: wp.vec3,
    ww: wp.vec3,
    tanh: float,
    aspect: float,
    ld: wp.vec3,
    hh: float,
    rmax: float,
    tilt: float,
    wob: float,
):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * aspect * tanh
    v = (2.0 * (float(height_px - 1 - i) + 0.5) / float(height_px) - 1.0) * tanh
    rd = wp.normalize(ww + u * uu + v * vv)

    vy = float(i) / float(height_px)
    bg = wp.vec3(0.020, 0.023, 0.034) * (1.0 - 0.6 * vy) + wp.vec3(0.006, 0.004, 0.010)

    t = float(0.0)
    hit = int(0)
    for _ in range(_STEPS):
        p = ro + rd * t
        d = _map(p, hh, rmax, tilt, wob)
        if d < 0.0009 * t + 0.0004:
            hit = 1
            break
        t += d * 0.72
        if t > _MAXD:
            break

    if hit == 0:
        img[i, j] = bg
        return

    p = ro + rd * t
    n = _normal(p, hh, rmax, tilt, wob)
    sh = _soft_shadow(p + n * 0.02, ld, hh, rmax, tilt, wob)
    ao = _ao(p, n, hh, rmax, tilt, wob)

    # material: a stained-chromosome violet with darker G-bands running across the arms (radial from the
    # centromere), a touch brighter toward the telomere tips. No cartoon colours — a specimen look.
    radial = wp.sqrt(p[0] * p[0] + p[1] * p[1])
    gb = 0.5 + 0.5 * wp.sin(radial * 4.6 + 0.4)
    dark = wp.vec3(0.26, 0.17, 0.42)
    lite = wp.vec3(0.74, 0.58, 0.86)
    albedo = wp.lerp(dark, lite, gb)
    tip = wp.clamp((radial - hh * 0.72) / (hh * 0.45), 0.0, 1.0)
    albedo = wp.lerp(albedo, wp.vec3(0.90, 0.82, 0.95), 0.35 * tip)

    v_dir = -rd
    lcol = wp.vec3(1.0, 0.96, 0.90)
    direct = shade_pbr(n, v_dir, ld, albedo, 0.44, 0.0, lcol) * (3.0 * sh)
    amb = wp.cw_mul(wp.vec3(0.30, 0.33, 0.45), albedo) * (0.36 * ao)
    sss = albedo * (0.14 * wp.clamp(wp.dot(n, ld) * 0.5 + 0.5, 0.0, 1.0))   # cheap waxy subsurface
    fres = wp.pow(wp.clamp(1.0 + wp.dot(rd, n), 0.0, 1.0), 3.0)
    rim = wp.vec3(0.45, 0.50, 0.72) * (0.32 * fres * ao)
    img[i, j] = direct + amb + sss + rim


def _condense(time: float):
    # phase 1 (condense a wavy thread into one chromatid), then phase 2 (the sister splays into the X)
    c = min(max((float(time) - 0.4) / 3.2, 0.0), 1.0)
    c = c * c * (3.0 - 2.0 * c)
    x = min(max((float(time) - 3.8) / 2.2, 0.0), 1.0)
    x = x * x * (3.0 - 2.0 * x)
    hh = 4.6 * (1.0 - c) + 2.85 * c
    rmax = 0.30 * (1.0 - c) + 1.02 * c
    wob = 0.58 * (1.0 - c)
    tilt = np.radians(21.0) * x
    return hh, rmax, tilt, wob


def _camera(time: float):
    u = min(max((float(time) - 0.3) / 5.4, 0.0), 1.0)
    u = u * u * (3.0 - 2.0 * u)
    dist = 17.0 * (1.0 - u) + 13.8 * u
    target = np.array([0.0, 0.0, 0.0], np.float32)
    direction = np.array([0.5, 0.14, 1.0], np.float32)
    direction = direction / np.linalg.norm(direction)
    ro = target + dist * direction
    ww = target - ro
    ww = ww / np.linalg.norm(ww)
    uu = np.cross(ww, np.array([0.0, 1.0, 0.0], np.float32))
    uu = uu / np.linalg.norm(uu)
    vv = np.cross(uu, ww)
    return ro, uu, vv, ww


def _render(width, height, time, mouse, device):
    W, H = int(width), int(height)
    hh, rmax, tilt, wob = _condense(float(time))
    ro, uu, vv, ww = _camera(float(time))
    ld = np.array([0.42, 0.72, 0.55], np.float32)
    ld = ld / np.linalg.norm(ld)
    tanh = float(np.tan(np.radians(32.0) * 0.5))
    aspect = W / float(H)

    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(
        _render_kernel,
        dim=(H, W),
        inputs=[img, W, H,
                wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
                wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]),
                tanh, aspect, wp.vec3(*[float(x) for x in ld]),
                float(hh), float(rmax), float(tilt), float(wob)],
        device=device,
    )
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=1.05, strength=0.32, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.2, preserve_hue=True)
    ldr = post.vignette(ldr, amount=0.3)
    return ldr


SCENE = Scene(
    name="warp_chromosome_solid",
    description=(
        "The chromosome as a solid, lit object — a ray-marched SDF (not a point cloud). The iconic metaphase "
        "X: two smooth sister chromatids joined at a pinched centromere, four banded arms with rounded "
        "telomere tips, lit like a stained specimen (key light, soft shadow, AO, PBR). It tells the story — a "
        "thin wavy chromatin thread condenses into one chromatid, then the sister splays open into the X."
    ),
    renderer=_render,
)
