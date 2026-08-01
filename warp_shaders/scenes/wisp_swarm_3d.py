"""The isochronous firework — the 3D ball un-explodes itself, twice per period.

Release motes from the center of the magic sphere in EVERY direction with EVERY
amplitude. In any Newtonian sky they would disperse forever. In the ball, each
mote follows the same exact closed form along its own ray —
``r(t) = r_max sin t/√(E²cos²t + sin²t)`` — and the period is 2π at EVERY
amplitude (``engine.wisp``, isochrony asserted). So the whole swarm passes back
through r = 0 SIMULTANEOUSLY every π: an explosion that un-explodes. The trap is
not just perfect for one wisp; it is perfect for ALL of them at once, forever.

What 3D adds is the size of what they fly through (all test-asserted):

* the equal-proper-distance shells ρ = 1..5 have areas ``A = 4π sinh²ρ`` —
  EXPONENTIAL growth (asserted A(ρ+1)/A(ρ) → e²), Euclidean 4πρ² only at small ρ;
* the ball volume is ``V = π(sinh 2ρ − 2ρ)`` — exactly ∫A dρ (asserted);
* and the skin theorem: ``V/A → 1/2`` (asserted) — however huge the ball,
  essentially all of its volume lies within ONE unit of the surface. Hyperbolic
  space is all skin and no core: the geometric seed of holography. The magenta
  ledger (the swarm's live V/A, doubled) rises toward its white asymptote line
  and never touches it.

Ledgers: cyan — the swarm's dispersion (breathes: spread, collapse, spread);
amber — the area of the swarm's mean shell (the exponential the motes climb);
magenta — 2·V/A against the white 1/2-asymptote line. --frames runs one cycle
(two collapses); iMouse pans. See ``docs/research/57-the-wisp-in-the-box.md``
(Stage 1 in 3D).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.wisp import (
    disk_radius,
    radial_geodesic_closed,
    sphere_area,
    volume_area_ratio,
)
from ..scene import Scene

_T_CYCLE = 16.0
_R_BUBBLE = 1.10
_N_MOTES = 40
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def _directions():
    dirs = []
    for k in range(_N_MOTES):
        z = 1.0 - 2.0 * (float(k) + 0.5) / float(_N_MOTES)
        rxy = math.sqrt(max(1.0 - z * z, 0.0))
        ph = _GOLDEN * float(k)
        dirs.append((rxy * math.cos(ph), z, rxy * math.sin(ph)))
    return dirs


_DIRS = _directions()
_AMPS = [0.6 + 2.4 * math.fmod(0.6180339887 * float(k + 1), 1.0)
         for k in range(_N_MOTES)]


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2,
                   cam: wp.vec3, fwd: wp.vec3, rgt: wp.vec3, upv: wp.vec3,
                   motes: wp.array(dtype=wp.vec3), n_motes: int,
                   disp_frac: float, area_frac: float, skin_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6

    rd = wp.normalize(fwd + rgt * (x / 2.0) + upv * (y / 2.0))
    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the ball: rim silhouette + haze + equal-rho shells ----
    oc = -cam
    b = wp.dot(oc, rd)
    dc2 = wp.dot(oc, oc) - b * b
    dc = wp.sqrt(wp.max(dc2, 0.0))
    if b > 0.0:
        d_rim = wp.abs(dc - 1.10)
        col = col + wp.vec3(0.55, 0.45, 1.00) * (0.85 * wp.exp(-(d_rim * d_rim) / 0.00035))
        col = col + wp.vec3(0.30, 0.25, 0.60) * (0.22 * wp.exp(-(d_rim * d_rim) / 0.009))
        if dc < 1.10:
            col = col + wp.vec3(0.10, 0.09, 0.22) * (0.16 * (1.10 - dc))
        for q in range(1, 6):
            rq = wp.tanh(0.5 * float(q)) * 1.10
            d_q = wp.abs(dc - rq)
            col = col + wp.vec3(0.14, 0.17, 0.30) * (0.5 * wp.exp(-(d_q * d_q) / 0.00016))

    # ---- the swarm (near bright, far dim: the fireworks read in depth) ----
    for m in range(n_motes):
        p = motes[m] - cam
        tc = wp.dot(p, rd)
        if tc > 0.0:
            perp2 = wp.dot(p, p) - tc * tc
            att = wp.min(wp.max(11.0 / wp.dot(p, p), 0.45), 1.8)
            col = col + wp.vec3(0.95, 0.82, 0.45) * (0.85 * att * wp.exp(-perp2 / 0.0006))
            col = col + wp.vec3(0.80, 0.60, 0.30) * (0.20 * att * wp.exp(-perp2 / 0.005))

    # ---- the ledgers: dispersion / shell area / the skin theorem ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * disp_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * area_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * skin_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0
    if x > 1.58 and x < 1.72 and wp.abs(y - (-1.05 + 2.0)) < 0.007:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 1.1      # V/A = 1/2: the asymptote

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    tc = 2.0 * math.pi * tau / _T_CYCLE     # coordinate time: collapses at 0 and pi

    # every mote on its own exact radial geodesic — one shared clock
    motes = []
    rho_abs = []
    for k in range(_N_MOTES):
        r_m = radial_geodesic_closed(math.sinh(_AMPS[k]), tc)
        rho = math.copysign(math.asinh(abs(r_m)), r_m)
        rho_abs.append(abs(rho))
        rr = disk_radius(abs(rho)) * _R_BUBBLE
        sgn = 1.0 if rho >= 0.0 else -1.0
        d = _DIRS[k]
        motes.append((sgn * rr * d[0], sgn * rr * d[1], sgn * rr * d[2]))

    rho_mean = sum(rho_abs) / float(_N_MOTES)
    disp_frac = min((sum(disk_radius(r) for r in rho_abs) / float(_N_MOTES)) / 0.85, 1.0)
    area_frac = min(sphere_area(rho_mean) / sphere_area(1.9), 1.0)
    skin_frac = 2.0 * volume_area_ratio(rho_mean) if rho_mean > 1e-9 else 0.0

    az = 2.0 * math.pi * (t / _T_CYCLE) + 2.1
    cam = np.array([3.3 * math.cos(az), 1.15, 3.3 * math.sin(az)])
    fwd = -cam / np.linalg.norm(cam)
    rgt = np.cross(fwd, [0.0, 1.0, 0.0])
    rgt = rgt / np.linalg.norm(rgt)
    upv = np.cross(rgt, fwd)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])),
                      wp.vec3(float(fwd[0]), float(fwd[1]), float(fwd[2])),
                      wp.vec3(float(rgt[0]), float(rgt[1]), float(rgt[2])),
                      wp.vec3(float(upv[0]), float(upv[1]), float(upv[2])),
                      wp.array(np.asarray(motes, np.float32), dtype=wp.vec3, device=device),
                      int(_N_MOTES),
                      float(disp_frac), float(area_frac), float(skin_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_swarm_3d",
    description="the isochronous firework: 40 motes released from the center of "
                "the magic sphere in every direction with every amplitude, each "
                "on its own exact closed-form radial geodesic — and because the "
                "period is 2pi at EVERY amplitude (isochrony asserted), the whole "
                "swarm passes back through r = 0 SIMULTANEOUSLY every pi: an "
                "explosion that un-explodes, twice per cycle, while the camera "
                "orbits the ball. The shells they fly through have areas "
                "4 pi sinh^2 rho (exponential, asserted; amber ledger), and the "
                "skin theorem rules the magenta ledger: V/A -> 1/2 (asserted) — "
                "all of hyperbolic space's volume lives within one unit of its "
                "surface, so the doubled ratio rises toward its white asymptote "
                "line and never touches it. Cyan ledger: the swarm's dispersion, "
                "breathing. --frames runs one cycle (two collapses).",
    renderer=_render,
)
