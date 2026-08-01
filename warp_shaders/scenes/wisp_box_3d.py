"""The wisp in the box, in 3D — the magic circle becomes a magic sphere.

"imagine you are a wisp in a box, the box is the boundary of simulation, you can
never reach those corner because of AdS/CFT since you are in a sphererical magic
circle / bubble." Stage 1, now genuinely spherical: the Poincaré BALL floats
inside the 3D simulation box, the camera orbits it, and the corner brackets hang
in depth — visible from every angle, reachable from none.

Every law is the SAME law as the flat scene, because the radial equation never
mentioned dimension (``engine.wisp``, all test-asserted):

* the rim of the ball sits at finite map radius and INFINITE proper distance
  (``ρ = 2·atanh r`` diverges — asserted);
* **coast** — the exact closed-form geodesic ``r(t) = r_max sin t/√(E²cos²t+sin²t)``
  sweeps the wisp through the center of the ball, period 2π at EVERY amplitude
  (isochrony asserted); free flight is planar (angular momentum conservation), so
  the 2D closed form applies verbatim in the wisp's own flight plane — here a
  tilted plane you look onto from outside;
* **burn** — the drive lights and proper distance climbs steadily while the map
  compresses as ``tanh(ρ/2)``: the wisp stalls visibly just inside the glowing
  sphere at full thrust (amber map-radius ledger pinned under its violet rim
  line) while the cyan proper-distance ledger keeps climbing, and the magenta
  reserve drains on the ``cosh ρ`` cliff (divergence asserted);
* **fall** — the reserve empties, the drive cuts, and the ball collects: the wisp
  whips back through the center of the sphere.

The nested translucent shells are the equal-proper-distance spheres ρ = 1..5,
crowding toward the rim — in 3D their AREAS grow as ``4π sinh²ρ`` (exponential,
asserted): each shell the wisp passes is e² larger than the one two units below.
--frames runs one cycle; iMouse pans. See
``docs/research/57-the-wisp-in-the-box.md`` (Stage 1 in 3D).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.wisp import (
    climb_energy,
    disk_radius,
    radial_geodesic_closed,
)
from ..scene import Scene

_T_CYCLE = 16.0
_T_COAST = 2.0 * math.pi
_T_BURN_END = 12.0
_RHO_COAST = 1.6
_CLIMB_RATE = 1.35
_R_BUBBLE = 1.10
_N_TRAIL = 16
_RHO_MAX = (_T_BURN_END - _T_COAST) * _CLIMB_RATE

# the flight plane: a tilted plane through the center of the ball
_N_PLANE = np.array([0.32, 1.0, 0.18])
_N_PLANE = _N_PLANE / np.linalg.norm(_N_PLANE)
_U_PLANE = np.cross([0.0, 1.0, 0.0], _N_PLANE)
_U_PLANE = _U_PLANE / np.linalg.norm(_U_PLANE)
_V_PLANE = np.cross(_N_PLANE, _U_PLANE)

# the box: corner brackets in depth (3 inward ticks x 3 dots per corner)
_BOX = np.array([1.55, 1.25, 1.55])


def _bracket_points():
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                c = np.array([sx, sy, sz]) * _BOX
                for axis in range(3):
                    a_hat = np.zeros(3)
                    a_hat[axis] = -np.sign(c[axis])
                    for m in range(3):
                        pts.append(c + a_hat * (0.05 + 0.115 * float(m)))
    return np.asarray(pts, np.float32)


_BRACKETS = _bracket_points()


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2,
                   cam: wp.vec3, fwd: wp.vec3, rgt: wp.vec3, upv: wp.vec3,
                   wpos: wp.vec3, fpos: wp.vec3, flame: float,
                   trail: wp.array(dtype=wp.vec4), n_trail: int,
                   brackets: wp.array(dtype=wp.vec3), n_brk: int,
                   rho_frac: float, disk_frac: float, res_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6

    rd = wp.normalize(fwd + rgt * (x / 2.0) + upv * (y / 2.0))
    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the ball: rim silhouette + interior haze + equal-rho shells ----
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

    # ---- the box: corner brackets hanging in depth (near bright, far dim) ----
    for m in range(n_brk):
        p = brackets[m] - cam
        tc = wp.dot(p, rd)
        if tc > 0.0:
            perp2 = wp.dot(p, p) - tc * tc
            att = wp.min(wp.max(12.0 / wp.dot(p, p), 0.45), 1.8)
            col = col + wp.vec3(0.45, 0.50, 0.60) * (0.7 * att * wp.exp(-perp2 / 0.0005))

    # ---- the trail: the wisp's recent past, fading ----
    for k in range(n_trail):
        tp = wp.vec3(trail[k][0], trail[k][1], trail[k][2]) - cam
        tc2 = wp.dot(tp, rd)
        if tc2 > 0.0:
            perp2 = wp.dot(tp, tp) - tc2 * tc2
            att = wp.min(wp.max(12.0 / wp.dot(tp, tp), 0.45), 1.8)
            col = col + wp.vec3(0.55, 0.85, 0.95) * \
                (0.5 * att * trail[k][3] * wp.exp(-perp2 / 0.0007))

    # ---- the drive flame: exhaust toward the center of the ball ----
    if flame > 0.0:
        fp = fpos - cam
        tcf = wp.dot(fp, rd)
        if tcf > 0.0:
            perp2 = wp.dot(fp, fp) - tcf * tcf
            att = wp.min(wp.max(12.0 / wp.dot(fp, fp), 0.45), 1.8)
            col = col + wp.vec3(1.00, 0.60, 0.20) * \
                (1.5 * att * flame * wp.exp(-perp2 / 0.004))

    # ---- the wisp ----
    wpr = wpos - cam
    tcw = wp.dot(wpr, rd)
    if tcw > 0.0:
        perp2 = wp.dot(wpr, wpr) - tcw * tcw
        att = wp.min(wp.max(12.0 / wp.dot(wpr, wpr), 0.45), 1.8)
        col = col + wp.vec3(0.80, 0.95, 1.00) * (2.0 * att * wp.exp(-perp2 / 0.0011))
        col = col + wp.vec3(0.40, 0.70, 1.00) * (0.5 * att * wp.exp(-perp2 / 0.010))

    # ---- the ledgers (screen space): rho / map radius + rim line / reserve ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * rho_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * disk_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.48 and x < 1.62 and wp.abs(y - (-1.05 + 2.0)) < 0.007:
        col = col + wp.vec3(0.55, 0.45, 1.00) * 1.2
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * res_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _rho_signed(tau: float) -> float:
    """Same exact stage-1 trajectory as the flat scene — dimension-blind."""
    if tau < _T_COAST:
        r = radial_geodesic_closed(math.sinh(_RHO_COAST), tau)
        return math.copysign(math.asinh(abs(r)), r)
    if tau < _T_BURN_END:
        return (tau - _T_COAST) * _CLIMB_RATE
    r_fall = math.sinh(_RHO_MAX)
    r = radial_geodesic_closed(r_fall, 0.5 * math.pi + (tau - _T_BURN_END))
    return math.copysign(math.asinh(abs(r)), r)


def _pos3(tau: float, t_abs: float):
    rho = _rho_signed(tau)
    phi = 0.55 + 0.055 * t_abs
    rr = disk_radius(abs(rho)) * _R_BUBBLE
    sgn = 1.0 if rho >= 0.0 else -1.0
    p = sgn * rr * (math.cos(phi) * _U_PLANE + math.sin(phi) * _V_PLANE)
    return p, rho


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    wpos, rho = _pos3(tau, t)

    # orbiting camera: one full lap per cycle (the GIF loops seamlessly)
    az = 2.0 * math.pi * (t / _T_CYCLE) + 0.6
    cam = np.array([3.4 * math.cos(az), 1.35, 3.4 * math.sin(az)])
    fwd = -cam / np.linalg.norm(cam)
    rgt = np.cross(fwd, [0.0, 1.0, 0.0])
    rgt = rgt / np.linalg.norm(rgt)
    upv = np.cross(rgt, fwd)

    trail = []
    for k in range(1, _N_TRAIL + 1):
        tk = t - 0.09 * float(k)
        pk, _ = _pos3(math.fmod(tk, _T_CYCLE) if tk >= 0.0 else 0.0, max(tk, 0.0))
        trail.append((pk[0], pk[1], pk[2], math.exp(-float(k) / 7.0)))

    burning = _T_COAST <= tau < _T_BURN_END
    flame = 1.0 if burning else 0.0
    nrm = float(np.linalg.norm(wpos))
    fdir = -wpos / nrm if nrm > 1e-6 else np.zeros(3)
    fpos = wpos + 0.10 * fdir

    rho_frac = min(abs(rho) / _RHO_MAX, 1.0)
    disk_frac = disk_radius(abs(rho))
    e_total = climb_energy(0.0, _RHO_MAX)
    if tau < _T_COAST:
        res_frac = 1.0
    elif burning:
        res_frac = max(1.0 - climb_energy(0.0, abs(rho)) / e_total, 0.0)
    else:
        res_frac = 0.02

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.vec3(float(cam[0]), float(cam[1]), float(cam[2])),
                      wp.vec3(float(fwd[0]), float(fwd[1]), float(fwd[2])),
                      wp.vec3(float(rgt[0]), float(rgt[1]), float(rgt[2])),
                      wp.vec3(float(upv[0]), float(upv[1]), float(upv[2])),
                      wp.vec3(float(wpos[0]), float(wpos[1]), float(wpos[2])),
                      wp.vec3(float(fpos[0]), float(fpos[1]), float(fpos[2])),
                      float(flame),
                      wp.array(np.asarray(trail, np.float32), dtype=wp.vec4, device=device),
                      int(_N_TRAIL),
                      wp.array(_BRACKETS, dtype=wp.vec3, device=device),
                      int(len(_BRACKETS)),
                      float(rho_frac), float(disk_frac), float(res_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_box_3d",
    description="stage 1 in 3D: the magic circle becomes a magic SPHERE — the "
                "Poincare ball floating inside the 3D simulation box, corner "
                "brackets hanging in depth, camera orbiting once per cycle. The "
                "same exact laws, because the radial equation never mentioned "
                "dimension: COAST on the closed-form geodesic through the center "
                "of the ball (period 2pi at every amplitude, isochrony asserted; "
                "free flight is planar so the 2D closed form applies verbatim in "
                "the tilted flight plane), BURN outward stalling just inside the "
                "glowing rim (amber map-radius ledger pinned under its violet rim "
                "line; magenta reserve draining on the cosh rho cliff, asserted), "
                "FALL back through the center when the fuel wall wins. The nested "
                "translucent shells are the equal-proper-distance spheres rho = "
                "1..5 whose areas grow as 4 pi sinh^2 rho (exponential, "
                "asserted). --frames runs one cycle.",
    renderer=_render,
)
