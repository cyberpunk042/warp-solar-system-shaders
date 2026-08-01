"""The wisp navigates the ball — stage 3 in 3D: the lens is now a sphere-full.

Stage 3 replayed inside the Poincaré BALL, and the geodesic lens finally shows
its true dimension. In the flat scene the mote fan lived in one plane; here the
motes are launched in DIFFERENT PLANES through the release point — every geodesic
lies in its own plane through the center (angular momentum conservation), so the
fan blossoms into a genuinely 3D flower — and the bubble still folds ALL of it
back to the single antipodal point at t = π and home at t = 2π (``engine.wisp``,
refocus asserted). The lens is not a trick of the plane; it is the geometry.

The rest of the trip is the flat scene's exact algebra (all test-asserted):

* the transfer arc between shells: ``E = cosh ρ_A cosh ρ_B``,
  ``L = sinh ρ_A sinh ρ_B`` (asserted at both apsides);
* the fare: total ``cosh²ρ_B − cosh²ρ_A``, path-independent (asserted);
* the isochronous subway: EVERY transfer takes Δt = π/2 and sweeps Δφ = π/2
  (asserted) — quarter period, quarter turn, in any plane, at any depth.

One 16-second cycle: orbit shell A → BOOST (amber spike; reserve pays the exact
bill) → coast the dotted quarter-turn arc → CIRCULARIZE → orbit shell B → release
the 3D mote flower and watch the ball refocus it — violet halo at the antipode at
t = π, green halo at home at t = 2π, the body arriving in step because its own
orbit is one more geodesic through the release point. Camera orbits once per
cycle. --frames runs one cycle; iMouse pans. See
``docs/research/57-the-wisp-in-the-box.md`` (Stage 3 + Stage 1 in 3D).
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
_N_MOTES = 12
_N_ROUTE = 30

# the body's orbital plane
_N_PLANE = np.array([0.30, 1.0, 0.16])
_N_PLANE = _N_PLANE / np.linalg.norm(_N_PLANE)
_U_PLANE = np.cross([0.0, 1.0, 0.0], _N_PLANE)
_U_PLANE = _U_PLANE / np.linalg.norm(_U_PLANE)
_V_PLANE = np.cross(_N_PLANE, _U_PLANE)


def _to_disk(r_metric: float) -> float:
    return math.tanh(0.5 * math.asinh(r_metric)) * _R_BUBBLE


_CACHE = None


def _tables():
    """The exact transfer arc + the lens fan's (r, dphi) trajectories."""
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
        lk = 0.6 + 0.42 * float(k)
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
                   mouse: wp.vec2,
                   cam: wp.vec3, fwd: wp.vec3, rgt: wp.vec3, upv: wp.vec3,
                   wpos: wp.vec3, fpos: wp.vec3, flame: float,
                   shell_a: float, shell_b: float,
                   route: wp.array(dtype=wp.vec3), n_route: int, route_glow: float,
                   motes: wp.array(dtype=wp.vec3), n_motes: int, mote_glow: float,
                   apos: wp.vec3, ping_a: float,
                   hpos: wp.vec3, ping_h: float,
                   rho_frac: float, thrust_frac: float, res_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6

    rd = wp.normalize(fwd + rgt * (x / 2.0) + upv * (y / 2.0))
    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the ball: rim + haze + equal-rho shells + the two working shells ----
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
            col = col + wp.vec3(0.14, 0.17, 0.30) * (0.35 * wp.exp(-(d_q * d_q) / 0.00016))
        d_sa = wp.abs(dc - shell_a)
        col = col + wp.vec3(0.30, 0.55, 0.65) * (0.30 * wp.exp(-(d_sa * d_sa) / 0.00018))
        d_sb = wp.abs(dc - shell_b)
        col = col + wp.vec3(0.30, 0.65, 0.55) * (0.40 * wp.exp(-(d_sb * d_sb) / 0.00022))

    # ---- the planned route: dotted transfer arc in the body's plane ----
    if route_glow > 0.0:
        for m in range(n_route):
            p = route[m] - cam
            tcr = wp.dot(p, rd)
            if tcr > 0.0:
                perp2 = wp.dot(p, p) - tcr * tcr
                col = col + wp.vec3(0.40, 0.75, 0.85) * \
                    (0.35 * route_glow * wp.exp(-perp2 / 0.0004))

    # ---- the geodesic lens: the 3D mote flower ----
    if mote_glow > 0.0:
        for m in range(n_motes):
            p = motes[m] - cam
            tcm = wp.dot(p, rd)
            if tcm > 0.0:
                perp2 = wp.dot(p, p) - tcm * tcm
                col = col + wp.vec3(0.95, 0.85, 0.50) * \
                    (0.8 * mote_glow * wp.exp(-perp2 / 0.0006))

    # ---- refocus halos: antipode at t = pi, home at t = 2 pi ----
    if ping_a > 0.001:
        pa = apos - cam
        tca = wp.dot(pa, rd)
        if tca > 0.0:
            dpa = wp.sqrt(wp.max(wp.dot(pa, pa) - tca * tca, 0.0))
            da = wp.abs(dpa - 0.11)
            col = col + wp.vec3(0.80, 0.70, 1.00) * (1.5 * ping_a * wp.exp(-(da * da) / 0.0005))
    if ping_h > 0.001:
        ph = hpos - cam
        tch = wp.dot(ph, rd)
        if tch > 0.0:
            dph = wp.sqrt(wp.max(wp.dot(ph, ph) - tch * tch, 0.0))
            dh = wp.abs(dph - 0.11)
            col = col + wp.vec3(0.70, 1.00, 0.85) * (1.5 * ping_h * wp.exp(-(dh * dh) / 0.0005))

    # ---- the drive flame (boost / circularize flashes) ----
    if flame > 0.0:
        fp = fpos - cam
        tcf = wp.dot(fp, rd)
        if tcf > 0.0:
            perp2 = wp.dot(fp, fp) - tcf * tcf
            col = col + wp.vec3(1.00, 0.60, 0.20) * (1.5 * flame * wp.exp(-perp2 / 0.004))

    # ---- the body: core + full hull bubble (grown in stage 2) ----
    wpr = wpos - cam
    tcw = wp.dot(wpr, rd)
    if tcw > 0.0:
        perp2 = wp.dot(wpr, wpr) - tcw * tcw
        col = col + wp.vec3(0.80, 0.95, 1.00) * (1.9 * wp.exp(-perp2 / 0.0011))
        dpw = wp.sqrt(perp2)
        d_h = wp.abs(dpw - 0.075)
        col = col + wp.vec3(0.55, 0.90, 0.75) * (1.0 * wp.exp(-(d_h * d_h) / 0.00016))

    # ---- the ledgers: altitude / burn / reserve ----
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
    budget = total / 0.62
    phi_launch = _PHI0 + _T_BOOST
    phi_arr = phi_launch + 0.5 * math.pi
    phi_rel = phi_arr + (_T_LENS - _T_CIRC)
    r_b = math.sinh(_RHO_B)

    def in_plane(rr_disk, ph):
        return rr_disk * (math.cos(ph) * _U_PLANE + math.sin(ph) * _V_PLANE)

    # the release direction and its perpendicular frame (for the 3D fan planes)
    d_rel = math.cos(phi_rel) * _U_PLANE + math.sin(phi_rel) * _V_PLANE
    e_in = -math.sin(phi_rel) * _U_PLANE + math.cos(phi_rel) * _V_PLANE
    e_out = _N_PLANE

    route_glow, mote_glow = 0.0, 0.0
    ping_a, ping_h = 0.0, 0.0
    mote_pos = [(0.0, 0.0, 0.0)] * _N_MOTES
    rho_now = _RHO_A

    if tau < _T_BOOST:
        phi = _PHI0 + tau
        wpos = in_plane(_to_disk(math.sinh(_RHO_A)), phi)
        res = 0.95
    elif tau < _T_CIRC:
        tc = (tau - _T_BOOST) / (_T_CIRC - _T_BOOST) * (0.5 * math.pi)
        r_now = float(np.interp(tc, arc_t, arc_r))
        phi = phi_launch + float(np.interp(tc, arc_t, arc_p))
        wpos = in_plane(_to_disk(r_now), phi)
        rho_now = math.asinh(r_now)
        route_glow = 1.0
        res = 0.95 - (boost / budget) * _smooth((tau - _T_BOOST) / 0.35)
    elif tau < _T_LENS:
        phi = phi_arr + (tau - _T_CIRC)
        wpos = in_plane(_to_disk(r_b), phi)
        rho_now = _RHO_B
        res = 0.95 - boost / budget - (circ / budget) * _smooth((tau - _T_CIRC) / 0.35)
    else:
        tc = (tau - _T_LENS) / (_T_CYCLE - _T_LENS) * (2.0 * math.pi)
        phi = phi_rel + tc
        wpos = in_plane(_to_disk(r_b), phi)
        rho_now = _RHO_B
        mote_glow = 1.0
        for k, (mts, mrs, mphis) in enumerate(motes_tab):
            rm = float(np.interp(tc, mts, mrs))
            dph = float(np.interp(tc, mts, mphis))
            dm = _to_disk(rm)
            # each mote's geodesic lives in its OWN plane through the release ray
            psi = math.pi * float(k) / float(_N_MOTES)
            e_k = math.cos(psi) * e_in + math.sin(psi) * e_out
            p = dm * (math.cos(dph) * d_rel + math.sin(dph) * e_k)
            mote_pos[k] = (p[0], p[1], p[2])
        ping_a = math.exp(-((tc - math.pi) / 0.22) ** 2)
        ping_h = math.exp(-((tc - 2.0 * math.pi) / 0.22) ** 2)
        res = 0.95 - total / budget

    nrm = float(np.linalg.norm(wpos))
    fdir = -wpos / nrm if nrm > 1e-6 else np.zeros(3)
    flame = math.exp(-((tau - _T_BOOST) / 0.18) ** 2) + \
        math.exp(-((tau - _T_CIRC) / 0.18) ** 2)
    flame = min(1.0, flame)
    fpos = wpos + 0.10 * fdir

    route_pts = []
    for m in range(_N_ROUTE):
        tc_m = (float(m) + 0.5) / float(_N_ROUTE) * (0.5 * math.pi)
        rm = float(np.interp(tc_m, arc_t, arc_r))
        pm = phi_launch + float(np.interp(tc_m, arc_t, arc_p))
        route_pts.append(tuple(in_plane(_to_disk(rm), pm)))

    dap = _to_disk(r_b)
    apos = -dap * d_rel                      # the antipode: -d_rel in EVERY plane
    hpos = dap * d_rel

    az = 2.0 * math.pi * (t / _T_CYCLE) + 0.35
    cam = np.array([3.4 * math.cos(az), 1.25, 3.4 * math.sin(az)])
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
                      wp.vec3(float(wpos[0]), float(wpos[1]), float(wpos[2])),
                      wp.vec3(float(fpos[0]), float(fpos[1]), float(fpos[2])),
                      float(flame),
                      float(_to_disk(math.sinh(_RHO_A))), float(_to_disk(r_b)),
                      wp.array(np.asarray(route_pts, np.float32), dtype=wp.vec3,
                               device=device),
                      int(_N_ROUTE), float(route_glow),
                      wp.array(np.asarray(mote_pos, np.float32), dtype=wp.vec3,
                               device=device),
                      int(_N_MOTES), float(mote_glow),
                      wp.vec3(float(apos[0]), float(apos[1]), float(apos[2])),
                      float(ping_a),
                      wp.vec3(float(hpos[0]), float(hpos[1]), float(hpos[2])),
                      float(ping_h),
                      float(rho_now / 2.0), float(flame), float(max(res, 0.02))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_navigate_3d",
    description="stage 3 in 3D: the trip between hover shells replayed inside "
                "the Poincare ball (boost off shell A paying the exact "
                "cosh(rhoA)(cosh(rhoB)-cosh(rhoA)) bill, the dotted quarter-turn "
                "transfer arc with E = cosh cosh and L = sinh sinh asserted at "
                "both apsides, circularize at shell B, fare telescoping to "
                "cosh^2 - cosh^2 path-independent) — and then the geodesic lens "
                "in its TRUE dimension: 12 motes launched in DIFFERENT PLANES "
                "through the release point (every geodesic lies in its own "
                "plane through the center), blossoming into a 3D flower that "
                "the ball folds back to the single antipodal point at t = pi "
                "(violet halo) and home at t = 2pi (green halo), the body "
                "arriving in step (refocus asserted). Camera orbits once per "
                "cycle. --frames runs one cycle.",
    renderer=_render,
)
