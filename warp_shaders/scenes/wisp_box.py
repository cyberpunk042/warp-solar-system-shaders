"""The wisp in the box — trapped by geometry, not by walls.

A wisp lives inside a box — the boundary of the simulation — but it can never reach
the corners, because it floats inside a spherical magic circle: an AdS bubble. The
rim sits at finite MAP radius and infinite PROPER distance (``ρ = 2·atanh(r)``,
divergence asserted), so the corners stay visible and forever out of reach. The
scene plays one full cycle of the wisp's stage-1 life (``engine.wisp``, every law
exact and test-asserted):

* **coast** — engines cold, the wisp free-falls on the exact radial geodesic
  ``r(t) = r_max·sin t/√(E²cos²t + sin²t)`` (closed form, asserted against RK4):
  it sweeps through the center and the bubble hands it back — EVERY orbit takes
  exactly 2π regardless of amplitude (isochrony asserted): the trap is perfect
  and patient;
* **burn** — the drive lights (exhaust flame toward the center) and proper
  distance climbs steadily — but the map compresses as ``tanh(ρ/2)``: the wisp
  visibly stalls just inside the rim at FULL thrust while the cyan
  proper-distance ledger keeps rising — progress without arrival. The magenta
  **energy reserve** drains as ``cosh(ρ)`` (asserted divergent): flat at first,
  then the cliff — each further step costs exponentially more (the wisp that
  wants altitude must grow and retain energy);
* **fall** — the reserve empties, the drive cuts, and the geometry collects its
  due: released from rest, the wisp whips back through the center (fastest where
  the map is loosest — the hyperbolic signature).

The amber ledger shows disk radius saturating BELOW its rim line — the punchline
in one bar: distance grows without bound, arrival never comes. Hovering anywhere
would be cheap (``a = tanh ρ < c²/L``, bound asserted); LEAVING is infinitely
expensive. Stage 2 (growing into a body) and stage 3 (navigation) come later.
--frames runs one cycle; iMouse pans. See
``docs/research/57-the-wisp-in-the-box.md``.
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
_T_COAST = 2.0 * math.pi         # one exact free-fall period
_T_BURN_END = 12.0
_RHO_COAST = 1.6                 # coast amplitude (proper distance)
_CLIMB_RATE = 1.35               # proper distance per second under full burn
_R_BUBBLE = 1.10                 # the magic circle's map radius on screen
_N_TRAIL = 22
_RHO_MAX = (_T_BURN_END - _T_COAST) * _CLIMB_RATE


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, wx: float, wy: float,
                   trail: wp.array(dtype=wp.vec3), n_trail: int,
                   flx: float, fly: float, flame: float,
                   rho_frac: float, disk_frac: float, res_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the box: the boundary of the simulation (corner brackets) ----
    bx = 2.16
    by = 1.22
    wb = wp.max(0.006, 1.6 * px)
    ax = wp.abs(x)
    ay = wp.abs(y)
    on_edge_x = wp.abs(ax - bx) < wb and ay < by
    on_edge_y = wp.abs(ay - by) < wb and ax < bx
    near_corner = ax > bx - 0.34 and ay > by - 0.34
    if (on_edge_x or on_edge_y) and near_corner:
        col = col + wp.vec3(0.45, 0.50, 0.60) * 0.9      # bright brackets
    elif on_edge_x or on_edge_y:
        col = col + wp.vec3(0.16, 0.18, 0.24) * 0.5      # faint frame

    # ---- the magic circle: rim at finite map radius, infinite distance ----
    r_pix = wp.sqrt(x * x + y * y)
    d_rim = wp.abs(r_pix - 1.10)
    wr = wp.max(0.007, 1.8 * px)
    col = col + wp.vec3(0.55, 0.45, 1.00) * (0.9 * wp.exp(-(d_rim * d_rim) / (wr * wr)))
    col = col + wp.vec3(0.30, 0.25, 0.60) * (0.25 * wp.exp(-(d_rim * d_rim) / (wr * wr * 25.0)))

    # ---- equal-proper-distance rings: rho = 1..5 crowd toward the rim ----
    wg = wp.max(0.004, 1.2 * px)
    for q in range(1, 6):
        rq = wp.tanh(0.5 * float(q)) * 1.10
        d_q = wp.abs(r_pix - rq)
        col = col + wp.vec3(0.14, 0.17, 0.30) * (0.40 * wp.exp(-(d_q * d_q) / (wg * wg)))

    # ---- the trail: the wisp's recent past, fading ----
    for k in range(n_trail):
        d2t = (x - trail[k][0]) * (x - trail[k][0]) + (y - trail[k][1]) * (y - trail[k][1])
        wt = wp.max(0.010, 1.8 * px)
        col = col + wp.vec3(0.55, 0.85, 0.95) * \
            (0.55 * trail[k][2] * wp.exp(-d2t / (wt * wt)))

    # ---- the drive flame: exhaust toward the center during burn ----
    if flame > 0.0:
        ex = wx + flx * 0.10
        ey = wy + fly * 0.10
        d2f = (x - ex) * (x - ex) + (y - ey) * (y - ey)
        wf = wp.max(0.030, 4.0 * px)
        col = col + wp.vec3(1.00, 0.60, 0.20) * (1.6 * flame * wp.exp(-d2f / (wf * wf)))

    # ---- the wisp ----
    d2w = (x - wx) * (x - wx) + (y - wy) * (y - wy)
    ww = wp.max(0.020, 2.8 * px)
    col = col + wp.vec3(0.80, 0.95, 1.00) * (2.0 * wp.exp(-d2w / (ww * ww)))
    col = col + wp.vec3(0.40, 0.70, 1.00) * (0.5 * wp.exp(-d2w / (ww * ww * 9.0)))

    # ---- the ledgers: proper distance / map radius (capped by the rim) / reserve ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * rho_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * disk_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.48 and x < 1.62 and wp.abs(y - (-1.05 + 2.0)) < 0.007:
        col = col + wp.vec3(0.55, 0.45, 1.00) * 1.2      # the rim line it never touches
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * res_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _rho_signed(tau: float) -> float:
    """The wisp's signed proper distance over one cycle (host-side): coast
    (exact closed-form geodesic), burn (steady climb), fall (released at rest,
    the geometry collects)."""
    if tau < _T_COAST:
        r = radial_geodesic_closed(math.sinh(_RHO_COAST), tau)
        return math.copysign(math.asinh(abs(r)), r)
    if tau < _T_BURN_END:
        return (tau - _T_COAST) * _CLIMB_RATE
    r_fall = math.sinh(_RHO_MAX)
    r = radial_geodesic_closed(r_fall, 0.5 * math.pi + (tau - _T_BURN_END))
    return math.copysign(math.asinh(abs(r)), r)


def _pos(tau: float, t_abs: float):
    rho = _rho_signed(tau)
    phi = 0.55 + 0.055 * t_abs
    rd = disk_radius(abs(rho)) * _R_BUBBLE
    sgn = 1.0 if rho >= 0.0 else -1.0
    return (sgn * rd * math.cos(phi), sgn * rd * math.sin(phi), rho)


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    wx, wy, rho = _pos(tau, t)

    # trail from the same deterministic trajectory
    trail = []
    for k in range(1, _N_TRAIL + 1):
        tk = t - 0.09 * float(k)
        xk, yk, _ = _pos(math.fmod(tk, _T_CYCLE) if tk >= 0.0 else 0.0, max(tk, 0.0))
        trail.append((xk, yk, math.exp(-float(k) / 9.0)))

    burning = _T_COAST <= tau < _T_BURN_END
    flame = 1.0 if burning else 0.0
    # exhaust points back toward the center
    nrm = math.hypot(wx, wy)
    flx, fly = (-wx / nrm, -wy / nrm) if nrm > 1e-6 else (0.0, 0.0)

    # ledgers
    rho_frac = min(abs(rho) / _RHO_MAX, 1.0)
    disk_frac = disk_radius(abs(rho))                     # capped by 1: the rim line
    e_total = climb_energy(0.0, _RHO_MAX)
    if tau < _T_COAST:
        res_frac = 1.0                                    # retained, ready
    elif burning:
        res_frac = max(1.0 - climb_energy(0.0, abs(rho)) / e_total, 0.0)
    else:
        res_frac = 0.02                                   # spent

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(wx), float(wy),
                      wp.array(np.asarray(trail, np.float32), dtype=wp.vec3, device=device),
                      int(_N_TRAIL),
                      float(flx), float(fly), float(flame),
                      float(rho_frac), float(disk_frac), float(res_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_box",
    description="a wisp in the simulation box that can never reach the corners: "
                "it lives in an AdS bubble whose rim is at finite map radius but "
                "infinite proper distance (2 atanh r, divergence asserted). One "
                "cycle of its stage-1 life: COAST on the exact closed-form "
                "geodesic (period 2pi at every amplitude — isochrony asserted: "
                "the trap always hands it back), BURN outward with the drive "
                "flame while the map compresses as tanh(rho/2) — the cyan "
                "proper-distance ledger climbs but the amber map-radius bar "
                "saturates below its rim line — and the magenta energy reserve "
                "drains on the cosh(rho) cliff (divergence asserted) until the "
                "drive cuts and the wisp FALLS back through the center. Hovering "
                "is cheap (a = tanh rho < 1, bound asserted); leaving is "
                "infinitely expensive. --frames runs one cycle.",
    renderer=_render,
)
