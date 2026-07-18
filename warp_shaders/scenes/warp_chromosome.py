"""Process 7 — the chromosome: the condensed metaphase X, as a solid lit specimen.

This is the culmination of the ladder. The earlier stages show the real conserved matter as solid strands
(the 30 nm fibre, the telomere t-loop). Here the fibre has folded all the way down — the ~50x higher-order
coil ``fold_chromatid`` computes — into the condensed chromosome. At this packing the coils are far below the
surface, so what you see is the **smooth envelope** of that folded matter: a metaphase chromosome, two sister
chromatids joined at the pinched centromere, four banded arms with rounded telomere caps. The envelope's
dimensions are taken straight from the fold lib (arm radius, half-height), so the surface is the real
condensed chromatid's outline — not a shape invented from nothing. It is ray-marched as a signed-distance
field with the engine's PBR (key light + soft self-shadow + AO), a stained specimen under the microscope.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pbr import shade_pbr
from ..genome.chromatid import fold_chromatid
from ..procedural.sdf import op_smooth_union
from ..scene import Scene

# envelope dimensions from the real fold (Process 7): the chromatid arm radius + half-height
_CH = fold_chromatid(sub=2, block=5)
_HH = wp.constant(float(_CH.height * 0.92))              # arm half-height
_RMAX = wp.constant(float(_CH.arm_radius * 0.36))       # surface radius of one arm (envelope of the coil)
_SEP = wp.constant(0.86)                                 # sisters bow apart along the arms

_STEPS = wp.constant(190)
_MAXD = wp.constant(80.0)


@wp.func
def _rprof(yn: float) -> float:
    # smooth radius profile along one chromatid arm: fat mid-arm, rounded tip, sharp centromere constriction
    a = wp.abs(yn)
    d = (a - 0.55) / 0.32
    prof = 0.34 + 0.68 * wp.exp(-d * d)
    cc = yn / 0.11
    prof = prof * (1.0 - 0.42 * wp.exp(-cc * cc))
    return prof


@wp.func
def _rod(p: wp.vec3, hh: float, rmax: float) -> float:
    y = p[1]
    cy = wp.clamp(y, -hh, hh)                            # clamp -> rounded caps
    rad = rmax * _rprof(cy / hh) * (1.0 + 0.020 * wp.sin(cy * 9.0))     # faint permanent surface banding
    return wp.sqrt(p[0] * p[0] + (y - cy) * (y - cy) + p[2] * p[2]) - rad


@wp.func
def _map(p: wp.vec3, hh: float, rmax: float, sep: float) -> float:
    gy = sep * wp.pow(wp.clamp(wp.abs(p[1]) / hh, 0.0, 1.0), 0.7)      # joined at centromere, bow apart on arms
    a = _rod(wp.vec3(p[0] - gy, p[1], p[2]), hh, rmax)
    b = _rod(wp.vec3(p[0] + gy, p[1], p[2]), hh, rmax)
    return op_smooth_union(a, b, 0.12)


@wp.func
def _normal(p: wp.vec3, hh: float, rmax: float, sep: float) -> wp.vec3:
    e = float(0.004)
    dx = _map(p + wp.vec3(e, 0.0, 0.0), hh, rmax, sep) - _map(p - wp.vec3(e, 0.0, 0.0), hh, rmax, sep)
    dy = _map(p + wp.vec3(0.0, e, 0.0), hh, rmax, sep) - _map(p - wp.vec3(0.0, e, 0.0), hh, rmax, sep)
    dz = _map(p + wp.vec3(0.0, 0.0, e), hh, rmax, sep) - _map(p - wp.vec3(0.0, 0.0, e), hh, rmax, sep)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _soft_shadow(p: wp.vec3, ld: wp.vec3, hh: float, rmax: float, sep: float) -> float:
    res = float(1.0)
    t = float(0.03)
    for _ in range(38):
        h = _map(p + ld * t, hh, rmax, sep)
        if h < 0.001:
            return 0.0
        res = wp.min(res, 12.0 * h / t)
        t += wp.clamp(h, 0.015, 0.25)
        if t > 14.0:
            break
    return wp.clamp(res, 0.0, 1.0)


@wp.func
def _ao(p: wp.vec3, n: wp.vec3, hh: float, rmax: float, sep: float) -> float:
    occ = float(0.0)
    sca = float(1.0)
    for k in range(5):
        hr = 0.02 + 0.11 * float(k)
        dd = _map(p + n * hr, hh, rmax, sep)
        occ += (hr - dd) * sca
        sca *= 0.82
    return wp.clamp(1.0 - 2.2 * occ, 0.0, 1.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height_px: int,
                   ro: wp.vec3, uu: wp.vec3, vv: wp.vec3, ww: wp.vec3, tanh: float, aspect: float,
                   ld: wp.vec3, hh: float, rmax: float, sep: float):
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
        d = _map(p, hh, rmax, sep)
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
    n = _normal(p, hh, rmax, sep)
    sh = _soft_shadow(p + n * 0.02, ld, hh, rmax, sep)
    ao = _ao(p, n, hh, rmax, sep)

    # stained-chromosome violet with transverse G-bands, brighter toward the telomere tips
    bc = wp.abs(p[1])
    raw = 0.60 * wp.sin(bc * 3.05 + 0.5) + 0.40 * wp.sin(bc * 1.73 + 2.1)
    gb = wp.clamp((raw * 0.5 + 0.5 - 0.36) / 0.28, 0.0, 1.0)
    dark = wp.vec3(0.24, 0.15, 0.40)
    lite = wp.vec3(0.78, 0.62, 0.88)
    albedo = wp.lerp(dark, lite, gb)
    tip = wp.clamp((bc - hh * 0.70) / (hh * 0.42), 0.0, 1.0)
    albedo = wp.lerp(albedo, wp.vec3(0.90, 0.83, 0.95), 0.30 * tip)

    v_dir = -rd
    lcol = wp.vec3(1.0, 0.96, 0.90)
    direct = shade_pbr(n, v_dir, ld, albedo, 0.44, 0.0, lcol) * (3.0 * sh)
    amb = wp.cw_mul(wp.vec3(0.30, 0.33, 0.45), albedo) * (0.36 * ao)
    sss = albedo * (0.14 * wp.clamp(wp.dot(n, ld) * 0.5 + 0.5, 0.0, 1.0))
    fres = wp.pow(wp.clamp(1.0 + wp.dot(rd, n), 0.0, 1.0), 3.0)
    rim = wp.vec3(0.45, 0.50, 0.72) * (0.32 * fres * ao)
    img[i, j] = direct + amb + sss + rim


def _camera(time: float):
    ang = 0.4 + 0.52 * float(time)                       # slow orbit (a turn over the gif)
    dist = 14.5
    target = np.array([0.0, 0.0, 0.0], np.float32)
    direction = np.array([0.5 * np.cos(ang), 0.14, 0.5 * np.sin(ang) + 0.75], np.float32)
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
    ro, uu, vv, ww = _camera(float(time))
    ld = np.array([0.42, 0.72, 0.55], np.float32)
    ld = ld / np.linalg.norm(ld)
    tanh = float(np.tan(np.radians(32.0) * 0.5))
    aspect = W / float(H)

    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(H, W),
              inputs=[img, W, H, wp.vec3(*[float(x) for x in ro]), wp.vec3(*[float(x) for x in uu]),
                      wp.vec3(*[float(x) for x in vv]), wp.vec3(*[float(x) for x in ww]),
                      tanh, aspect, wp.vec3(*[float(x) for x in ld]), _HH, _RMAX, _SEP],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    hdr = post.bloom(hdr, threshold=1.05, strength=0.32, radius=4, passes=2)
    ldr = post.tonemap(hdr, mode="aces", exposure=1.2, preserve_hue=True)
    return post.vignette(ldr, amount=0.3)


SCENE = Scene(
    name="warp_chromosome",
    description=(
        "Process 7 — the chromosome: the condensed metaphase X as a solid lit specimen. The fibre has folded "
        "all the way down (the ~50x higher-order coil of fold_chromatid); at that packing the coils are below "
        "the surface, so this is the smooth envelope of the real folded matter — two sister chromatids joined "
        "at the pinched centromere, four banded arms with rounded telomere caps, its arm radius and half-height "
        "taken from the fold lib. Ray-marched as an SDF with PBR + soft shadow + AO, a stained chromosome."
    ),
    renderer=_render,
)
