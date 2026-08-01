"""The wisp's shadow — the boundary sees everything, exactly.

"you can never reach those corner because of AdS/CFT since you are in a
sphererical magic circle / bubble." The AdS/CFT half of the brief, rendered: the
bubble has a boundary theory living on its rim, and the wisp casts an EXACT
shadow on it — the bulk-to-boundary propagator of global AdS₃,

    K(ρ, θ) = (cosh ρ − sinh ρ · cos θ)^(−Δ)

drawn live on the rim as the wisp flies its stage-1 cycle (``engine.wisp``,
every law test-asserted):

* **coast** — the wisp sweeps through the center on the exact geodesic and its
  shadow SLOSHES around the whole rim: at ρ = 0 the kernel is uniform (the
  boundary sees it everywhere at once), then it gathers on the near side, then
  smears through uniformity to the OPPOSITE side as the wisp crosses — the
  boundary watching a pendulum through a fisheye;
* **burn** — as proper distance climbs, the shadow SHARPENS exponentially:
  peak-to-antipode contrast ``e^{2Δρ}`` exactly (asserted at machine precision),
  half-max width ``θ_½ = acos[(cosh ρ − 2^{1/Δ}e^{−ρ})/sinh ρ]`` closed-form
  (asserted), shrinking as ``2√(2^{1/Δ}−1)·e^{−ρ}`` — the UV/IR correspondence:
  bulk depth IS boundary resolution;
* **and the conserved imprint** — the magenta ledger holds the LIVE integral
  ``∫K dθ``, and it NEVER MOVES: for Δ = 1 the total is 2π at every ρ
  (asserted at 10⁻⁶). Climbing concentrates the shadow but cannot change its
  total. The boundary never loses track of the wisp; holography is not
  surveillance added to the bubble — it is the bubble.

Ledgers: cyan — shadow width θ_½/π (SHRINKS under burn); amber — log-contrast
2Δρ (grows); magenta — the live ∫K dθ / 2π (FLAT, forever, under its white
conservation line). --frames runs one coast-burn-fall cycle; iMouse pans. See
``docs/research/57-the-wisp-in-the-box.md`` (The boundary sees everything).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.wisp import (
    climb_energy,
    disk_radius,
    radial_geodesic_closed,
    shadow_kernel,
    shadow_width,
)
from ..scene import Scene

_T_CYCLE = 16.0
_T_COAST = 2.0 * math.pi
_T_BURN_END = 12.0
_RHO_COAST = 1.6
_CLIMB_RATE = 1.35
_R_BUBBLE = 1.10
_N_TRAIL = 18
_RHO_MAX = (_T_BURN_END - _T_COAST) * _CLIMB_RATE
_DELTA = 1.0


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, wx: float, wy: float,
                   trail: wp.array(dtype=wp.vec3), n_trail: int,
                   flx: float, fly: float, flame: float,
                   rho_abs: float, th0: float, k_peak: float,
                   width_frac: float, contrast_frac: float, total_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the box: corner brackets (the simulation boundary) ----
    bx = 2.16
    by = 1.22
    wb = wp.max(0.006, 1.6 * px)
    ax = wp.abs(x)
    ay = wp.abs(y)
    on_edge_x = wp.abs(ax - bx) < wb and ay < by
    on_edge_y = wp.abs(ay - by) < wb and ax < bx
    near_corner = ax > bx - 0.34 and ay > by - 0.34
    if (on_edge_x or on_edge_y) and near_corner:
        col = col + wp.vec3(0.45, 0.50, 0.60) * 0.9
    elif on_edge_x or on_edge_y:
        col = col + wp.vec3(0.16, 0.18, 0.24) * 0.5

    # ---- the rim + the LIVE shadow: the exact propagator, painted on it ----
    r_pix = wp.sqrt(x * x + y * y)
    ang = wp.atan2(y, x)
    # normalized kernel K(rho, ang - th0) / K_peak in [e^{-2 rho}, 1]
    kern = wp.pow(wp.cosh(rho_abs) - wp.sinh(rho_abs) * wp.cos(ang - th0), -1.0) / k_peak
    d_rim = wp.abs(r_pix - 1.10)
    wr = wp.max(0.007, 1.8 * px)
    # base rim (faint violet) + the shadow's glow riding on it (warm, kernel-shaped)
    col = col + wp.vec3(0.40, 0.33, 0.75) * (0.45 * wp.exp(-(d_rim * d_rim) / (wr * wr)))
    shadow = wp.pow(kern, 0.55)
    col = col + wp.vec3(1.00, 0.78, 0.35) * \
        (1.35 * shadow * wp.exp(-(d_rim * d_rim) / (wr * wr)))
    # outer halo: the boundary theory lighting up beyond the rim
    if r_pix > 1.10:
        d_out = r_pix - 1.10
        col = col + wp.vec3(1.00, 0.70, 0.30) * (0.35 * shadow * wp.exp(-d_out / 0.10))

    # ---- equal-proper-distance rings ----
    wg = wp.max(0.004, 1.2 * px)
    for q in range(1, 6):
        rq = wp.tanh(0.5 * float(q)) * 1.10
        d_q = wp.abs(r_pix - rq)
        col = col + wp.vec3(0.14, 0.17, 0.30) * (0.35 * wp.exp(-(d_q * d_q) / (wg * wg)))

    # ---- the trail ----
    for k in range(n_trail):
        d2t = (x - trail[k][0]) * (x - trail[k][0]) + (y - trail[k][1]) * (y - trail[k][1])
        wt = wp.max(0.010, 1.8 * px)
        col = col + wp.vec3(0.55, 0.85, 0.95) * \
            (0.5 * trail[k][2] * wp.exp(-d2t / (wt * wt)))

    # ---- the drive flame ----
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

    # ---- the ledgers: width (shrinks) / log-contrast (grows) / total (FLAT) ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * width_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * contrast_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * total_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0
    if x > 1.58 and x < 1.72 and wp.abs(y - (-1.05 + 2.0 * 0.85)) < 0.007:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 1.1      # conservation: total = 2 pi

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _rho_signed(tau: float) -> float:
    """Same exact stage-1 trajectory as wisp_box."""
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
    return (sgn * rd * math.cos(phi), sgn * rd * math.sin(phi), rho, phi)


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    wx, wy, rho, phi = _pos(tau, t)
    rho_abs = abs(rho)
    th0 = phi if rho >= 0.0 else phi + math.pi     # the shadow peaks over the wisp

    trail = []
    for k in range(1, _N_TRAIL + 1):
        tk = t - 0.09 * float(k)
        xk, yk, _, _ = _pos(math.fmod(tk, _T_CYCLE) if tk >= 0.0 else 0.0, max(tk, 0.0))
        trail.append((xk, yk, math.exp(-float(k) / 8.0)))

    burning = _T_COAST <= tau < _T_BURN_END
    flame = 1.0 if burning else 0.0
    nrm = math.hypot(wx, wy)
    flx, fly = (-wx / nrm, -wy / nrm) if nrm > 1e-6 else (0.0, 0.0)

    # the three shadow laws, live
    k_peak = shadow_kernel(rho_abs, 0.0, _DELTA)
    width_frac = shadow_width(rho_abs, _DELTA) / math.pi
    contrast_frac = min(2.0 * _DELTA * rho_abs / (2.0 * _DELTA * _RHO_MAX), 1.0)
    n_int = 512                                     # the LIVE integral — it never moves
    total = sum(shadow_kernel(rho_abs, 2.0 * math.pi * (k + 0.5) / n_int, _DELTA)
                for k in range(n_int)) * 2.0 * math.pi / n_int
    total_frac = 0.85 * total / (2.0 * math.pi)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(wx), float(wy),
                      wp.array(np.asarray(trail, np.float32), dtype=wp.vec3, device=device),
                      int(_N_TRAIL),
                      float(flx), float(fly), float(flame),
                      float(rho_abs), float(th0), float(k_peak),
                      float(width_frac), float(contrast_frac), float(total_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_shadow",
    description="the AdS/CFT half of the brief: the wisp's EXACT boundary "
                "shadow — the bulk-to-boundary propagator K = (cosh rho - "
                "sinh rho cos theta)^(-Delta) painted live on the rim while the "
                "wisp flies its stage-1 cycle. Coasting, the shadow sloshes "
                "around the whole rim (uniform when the wisp crosses the "
                "center); burning, it sharpens exponentially — contrast "
                "e^(2 Delta rho) exact at machine precision (asserted), width "
                "acos-closed-form shrinking as 2 sqrt(2^(1/Delta)-1) e^(-rho) "
                "(asserted; the UV/IR correspondence: bulk depth IS boundary "
                "resolution) — and the magenta ledger holds the LIVE integral "
                "of the shadow, which NEVER MOVES: total = 2 pi at every rho "
                "(asserted). The boundary never loses track of the wisp; "
                "holography is not surveillance added to the bubble — it is "
                "the bubble. --frames runs one coast-burn-fall cycle.",
    renderer=_render,
)
