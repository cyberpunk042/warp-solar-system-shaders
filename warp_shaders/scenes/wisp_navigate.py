"""The wisp navigates — stage 3: boost, ride the arc, and trust the lens.

"For now we can focus on the fact that for you to move forward in the simulation
you have to use your drives, your engines." Stage 3 is getting SOMEWHERE: the
cost algebra of travel in a space that resists arrival (``engine.wisp``, every
law exact and test-asserted):

* **the transfer arc** — the ballistic route between hover shells ρ_A and ρ_B is
  the geodesic whose apsides ARE the two shells, with pure hyperbolic constants
  ``E = cosh ρ_A cosh ρ_B``, ``L = sinh ρ_A sinh ρ_B`` (asserted at both apsides);
* **the fare** — boosting off the ρ_A orbit costs ``cosh ρ_A·Δcosh``,
  circularizing at ρ_B costs ``cosh ρ_B·Δcosh``, and the total telescopes to
  ``cosh²ρ_B − cosh²ρ_A`` — EXACTLY the orbit-energy difference (asserted):
  the bill is path-independent. No clever route exists; only the fare;
* **the isochronous subway** — every transfer between ANY two shells takes
  coordinate time Δt = π/2 and sweeps Δφ = π/2 exactly (asserted): a quarter
  period, a quarter turn, near or far. The timetable is trivial; the fare is not;
* **the geodesic lens** — release test motes in ANY direction with ANY speed:
  ALL of them (and the body itself, whose circular orbit is one more member of
  the family) reconverge at the ANTIPODE at t = π and come HOME at t = 2π
  (asserted at 1e-6 for three different launches). In the bubble you cannot get
  lost — only be early with more fuel.

One 16-second cycle: orbit shell A → BOOST (amber spike; magenta reserve steps
down by the exact boost bill) → coast the quarter-turn arc along the dotted route
→ CIRCULARIZE (second spike, second step) → orbit shell B → release the mote fan
and watch the lens: spread, antipodal ping, home ping. --frames runs one cycle;
iMouse pans. See ``docs/research/57-the-wisp-in-the-box.md`` (Stage 3).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.wisp import (
    orbit_geodesic,
    transfer_cost,
    transfer_orbit,
)
from ..scene import Scene

_T_CYCLE = 16.0
_T_BOOST, _T_CIRC, _T_LENS = 2.5, 6.5, 9.5
_RHO_A, _RHO_B = 0.8, 1.6
_R_BUBBLE = 1.10
_PHI0 = 2.05
_N_MOTES = 10
_N_ROUTE = 36


def _to_disk(r_metric: float) -> float:
    """Global radius r = sinh(rho) -> Poincare-disk display radius."""
    return math.tanh(0.5 * math.asinh(r_metric)) * _R_BUBBLE


_CACHE = None


def _tables():
    """Precompute (once) the exact transfer arc and the lens-fan trajectories."""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    e, ell = transfer_orbit(_RHO_A, _RHO_B)
    r_a, r_b = math.sinh(_RHO_A), math.sinh(_RHO_B)
    ts, rs, phis = orbit_geodesic(e, ell, r_a + 1e-9, 0.0, 1.0,
                                  0.5 * math.pi, 6000)
    arc = (np.asarray(ts), np.asarray(rs), np.asarray(phis))
    motes = []
    u0 = r_b * r_b
    for k in range(_N_MOTES):
        lk = 0.6 + 0.5 * float(k)
        e_min = math.sqrt((1.0 + u0) * (1.0 + lk * lk / u0))
        ek = e_min * (1.05 + 0.03 * float(k % 3))
        sgn = 1.0 if k % 2 == 0 else -1.0
        mts, mrs, mphis = orbit_geodesic(ek, lk, r_b, 0.0, sgn,
                                         2.0 * math.pi, 6000)
        motes.append((np.asarray(mts), np.asarray(mrs), np.asarray(mphis)))
    _CACHE = (arc, motes)
    return _CACHE


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, wx: float, wy: float,
                   flx: float, fly: float, flame: float,
                   shell_a: float, shell_b: float,
                   route: wp.array(dtype=wp.vec2), n_route: int, route_glow: float,
                   motes: wp.array(dtype=wp.vec2), n_motes: int, mote_glow: float,
                   apx: float, apy: float, ping_a: float,
                   hpx: float, hpy: float, ping_h: float,
                   rho_frac: float, thrust_frac: float, res_frac: float):
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

    # ---- the magic circle + equal-distance rings ----
    r_pix = wp.sqrt(x * x + y * y)
    d_rim = wp.abs(r_pix - 1.10)
    wr = wp.max(0.007, 1.8 * px)
    col = col + wp.vec3(0.55, 0.45, 1.00) * (0.9 * wp.exp(-(d_rim * d_rim) / (wr * wr)))
    col = col + wp.vec3(0.30, 0.25, 0.60) * (0.25 * wp.exp(-(d_rim * d_rim) / (wr * wr * 25.0)))
    wg = wp.max(0.004, 1.2 * px)
    for q in range(1, 6):
        rq = wp.tanh(0.5 * float(q)) * 1.10
        d_q = wp.abs(r_pix - rq)
        col = col + wp.vec3(0.14, 0.17, 0.30) * (0.40 * wp.exp(-(d_q * d_q) / (wg * wg)))

    # ---- the two hover shells: origin and destination ----
    d_sa = wp.abs(r_pix - shell_a)
    col = col + wp.vec3(0.30, 0.55, 0.65) * (0.25 * wp.exp(-(d_sa * d_sa) / (wg * wg)))
    d_sb = wp.abs(r_pix - shell_b)
    col = col + wp.vec3(0.30, 0.65, 0.55) * (0.30 * wp.exp(-(d_sb * d_sb) / (wg * wg)))

    # ---- the planned route: dotted transfer arc (visible while riding it) ----
    if route_glow > 0.0:
        wq = wp.max(0.008, 1.4 * px)
        for m in range(n_route):
            p = route[m]
            d2q = (x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1])
            col = col + wp.vec3(0.40, 0.75, 0.85) * (0.35 * route_glow * wp.exp(-d2q / (wq * wq)))

    # ---- the geodesic lens: the mote fan ----
    if mote_glow > 0.0:
        wm = wp.max(0.012, 1.9 * px)
        for m in range(n_motes):
            p = motes[m]
            d2m = (x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1])
            col = col + wp.vec3(0.95, 0.85, 0.50) * (0.75 * mote_glow * wp.exp(-d2m / (wm * wm)))

    # ---- refocus pings: antipode at t=pi, home at t=2pi ----
    if ping_a > 0.001:
        d2a = (x - apx) * (x - apx) + (y - apy) * (y - apy)
        da = wp.abs(wp.sqrt(d2a) - 0.11)
        wpg = wp.max(0.012, 2.2 * px)
        col = col + wp.vec3(0.80, 0.70, 1.00) * (1.5 * ping_a * wp.exp(-(da * da) / (wpg * wpg)))
    if ping_h > 0.001:
        d2h = (x - hpx) * (x - hpx) + (y - hpy) * (y - hpy)
        dh = wp.abs(wp.sqrt(d2h) - 0.11)
        wph = wp.max(0.012, 2.2 * px)
        col = col + wp.vec3(0.70, 1.00, 0.85) * (1.5 * ping_h * wp.exp(-(dh * dh) / (wph * wph)))

    # ---- the drive flame (boost / circularize flashes only) ----
    if flame > 0.0:
        ex = wx + flx * 0.11
        ey = wy + fly * 0.11
        d2f = (x - ex) * (x - ex) + (y - ey) * (y - ey)
        wf = wp.max(0.032, 4.2 * px)
        col = col + wp.vec3(1.00, 0.60, 0.20) * (1.6 * flame * wp.exp(-d2f / (wf * wf)))

    # ---- the body: wisp core + full hull ring (grown in stage 2) ----
    dxw = x - wx
    dyw = y - wy
    d2w = dxw * dxw + dyw * dyw
    ww = wp.max(0.018, 2.6 * px)
    col = col + wp.vec3(0.80, 0.95, 1.00) * (1.9 * wp.exp(-d2w / (ww * ww)))
    dw = wp.sqrt(d2w)
    d_h = wp.abs(dw - 0.085)
    ang = wp.atan2(dyw, dxw)
    if wp.sin(ang * 8.0) > -0.25:
        wh = wp.max(0.006, 1.5 * px)
        col = col + wp.vec3(0.55, 0.90, 0.75) * (1.0 * wp.exp(-(d_h * d_h) / (wh * wh)))

    # ---- the ledgers: altitude rho / burn / reserve ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * rho_frac:
        col = col + wp.vec3(0.30, 0.75, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * thrust_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * res_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _smooth(a: float) -> float:
    a = max(0.0, min(1.0, a))
    return a * a * (3.0 - 2.0 * a)


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    (arc_t, arc_r, arc_p), motes_tab = _tables()

    boost, circ, total = transfer_cost(_RHO_A, _RHO_B)
    budget = total / 0.62                        # the trip spends 62% of the reserve
    phi_launch = _PHI0 + _T_BOOST                # omega = 1 on shell A
    phi_arr = phi_launch + 0.5 * math.pi         # every transfer sweeps pi/2 exactly
    phi_rel = phi_arr + (_T_LENS - _T_CIRC)      # omega = 1 on shell B until release
    r_b = math.sinh(_RHO_B)

    route_glow, mote_glow = 0.0, 0.0
    ping_a, ping_h = 0.0, 0.0
    mote_pos = [(0.0, 0.0)] * _N_MOTES
    rho_now = _RHO_A

    if tau < _T_BOOST:                           # orbit shell A, omega = 1
        phi = _PHI0 + tau
        rd = _to_disk(math.sinh(_RHO_A))
        res = 0.95
    elif tau < _T_CIRC:                          # the quarter-period arc
        tc = (tau - _T_BOOST) / (_T_CIRC - _T_BOOST) * (0.5 * math.pi)
        r_now = float(np.interp(tc, arc_t, arc_r))
        phi = phi_launch + float(np.interp(tc, arc_t, arc_p))
        rd = _to_disk(r_now)
        rho_now = math.asinh(r_now)
        route_glow = 1.0
        res = 0.95 - (boost / budget) * _smooth((tau - _T_BOOST) / 0.35)
    elif tau < _T_LENS:                          # orbit shell B, omega = 1
        phi = phi_arr + (tau - _T_CIRC)
        rd = _to_disk(r_b)
        rho_now = _RHO_B
        res = 0.95 - boost / budget - (circ / budget) * _smooth((tau - _T_CIRC) / 0.35)
    else:                                        # the geodesic lens
        tc = (tau - _T_LENS) / (_T_CYCLE - _T_LENS) * (2.0 * math.pi)
        phi = phi_rel + tc                       # the body is one more geodesic
        rd = _to_disk(r_b)
        rho_now = _RHO_B
        mote_glow = 1.0
        for k, (mts, mrs, mphis) in enumerate(motes_tab):
            rm = float(np.interp(tc, mts, mrs))
            pm = phi_rel + float(np.interp(tc, mts, mphis))
            dm = _to_disk(rm)
            mote_pos[k] = (dm * math.cos(pm), dm * math.sin(pm))
        ping_a = math.exp(-((tc - math.pi) / 0.22) ** 2)
        ping_h = math.exp(-((tc - 2.0 * math.pi) / 0.22) ** 2)
        res = 0.95 - total / budget

    wx, wy = rd * math.cos(phi), rd * math.sin(phi)

    # burn flashes at the two impulses; exhaust points backwards along the orbit
    flame = math.exp(-((tau - _T_BOOST) / 0.18) ** 2) + \
        math.exp(-((tau - _T_CIRC) / 0.18) ** 2)
    flame = min(1.0, flame)
    flx, fly = math.sin(phi), -math.cos(phi)     # retrograde exhaust: prograde boost

    # route markers along the exact transfer arc
    route_pts = []
    for m in range(_N_ROUTE):
        tc_m = (float(m) + 0.5) / float(_N_ROUTE) * (0.5 * math.pi)
        rm = float(np.interp(tc_m, arc_t, arc_r))
        pm = phi_launch + float(np.interp(tc_m, arc_t, arc_p))
        dm = _to_disk(rm)
        route_pts.append((dm * math.cos(pm), dm * math.sin(pm)))

    dap = _to_disk(r_b)
    apx, apy = dap * math.cos(phi_rel + math.pi), dap * math.sin(phi_rel + math.pi)
    hpx, hpy = dap * math.cos(phi_rel), dap * math.sin(phi_rel)

    route_arr = wp.array([wp.vec2(px_, py_) for px_, py_ in route_pts],
                         dtype=wp.vec2, device=device)
    mote_arr = wp.array([wp.vec2(px_, py_) for px_, py_ in mote_pos],
                        dtype=wp.vec2, device=device)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(wx), float(wy),
                      float(flx), float(fly), float(flame),
                      float(_to_disk(math.sinh(_RHO_A))), float(_to_disk(r_b)),
                      route_arr, _N_ROUTE, float(route_glow),
                      mote_arr, _N_MOTES, float(mote_glow),
                      float(apx), float(apy), float(ping_a),
                      float(hpx), float(hpy), float(ping_h),
                      float(rho_now / 2.0), float(flame), float(max(res, 0.02))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_navigate",
    description="stage 3: navigation — the body boosts off its shell-A orbit "
                "(amber spike; the reserve steps down by the exact "
                "cosh(rhoA)(cosh(rhoB)-cosh(rhoA)) boost bill), coasts the "
                "ballistic transfer arc whose constants are pure hyperbolic "
                "algebra (E = cosh(rhoA)cosh(rhoB), L = sinh(rhoA)sinh(rhoB), "
                "asserted at both apsides) — a transfer that takes Dt = pi/2 "
                "and sweeps Dphi = pi/2 EXACTLY between any two shells "
                "(asserted: the isochronous subway) — circularizes at shell B "
                "(second spike; total fare telescopes to cosh^2 - cosh^2, "
                "path-independent, asserted), and then releases a fan of test "
                "motes into the geodesic lens: every launch, any direction, any "
                "speed (and the body itself) reconverges at the ANTIPODE at "
                "t = pi and comes HOME at t = 2pi (asserted at 1e-6). In the "
                "bubble you cannot get lost. --frames runs one "
                "orbit-boost-coast-circularize-lens cycle.",
    renderer=_render,
)
