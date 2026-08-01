"""The wisp's body in 3D — grow, climb, hover on flame, orbit the sphere for free.

Stage 2 replayed inside the Poincaré BALL: the body assembles low in the 3D
bubble where mass is cheapest, climbs the cosh cliff on its drive, hovers at the
green shell — now a full translucent SPHERE it holds altitude against — and then
tips into a circular orbit and cuts the flame. Every law is the flat scene's law
(``engine.wisp``, all test-asserted), because the radial equation never mentioned
dimension and free flight is planar:

* the hull grows as a bubble around the core, low where ``cosh ρ ≈ 1``;
* the climb pays the exact ``cosh ρ₂ − cosh ρ₁`` bill from the reserve;
* hovering pins the amber thrust bar at ``a = tanh ρ_h < c²/L`` (bound asserted)
  just under its white engine-max line, while the reserve bleeds;
* the orbit sits at ``L = r²`` exactly (V′ = 0 asserted, stable V″ = 8 asserted),
  costs ``cosh²ρ`` once (asserted), needs zero thrust forever — and turns at
  ``ω = 1`` like EVERY circular orbit in the bubble (asserted): the companion
  mote circling a lower shell in a DIFFERENT part of the sphere stays in step,
  lap for lap, forever.

The camera orbits the ball once per cycle; the two shell-spheres (the hover shell
and the companion's) glow as nested translucent silhouettes. Ledgers: green body
growth, amber thrust with its white max line, magenta reserve. --frames runs one
grow-climb-hover-orbit cycle; iMouse pans. See
``docs/research/57-the-wisp-in-the-box.md`` (Stage 1 in 3D + Stage 2).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.wisp import (
    climb_energy,
    disk_radius,
    hover_acceleration,
)
from ..scene import Scene

_T_CYCLE = 16.0
_T_GROW, _T_CLIMB, _T_HOVER = 4.0, 7.0, 11.0
_RHO_LOW, _RHO_SHELL = 0.35, 1.5
_RHO_COMPANION = 0.7
_R_BUBBLE = 1.10
_PHI0 = 2.15

# the body's working plane and the companion's (different planes — 3D)
_N_BODY = np.array([0.30, 1.0, 0.16])
_N_BODY = _N_BODY / np.linalg.norm(_N_BODY)
_U_BODY = np.cross([0.0, 1.0, 0.0], _N_BODY)
_U_BODY = _U_BODY / np.linalg.norm(_U_BODY)
_V_BODY = np.cross(_N_BODY, _U_BODY)
_N_COMP = np.array([-0.45, 1.0, 0.35])
_N_COMP = _N_COMP / np.linalg.norm(_N_COMP)
_U_COMP = np.cross([0.0, 1.0, 0.0], _N_COMP)
_U_COMP = _U_COMP / np.linalg.norm(_U_COMP)
_V_COMP = np.cross(_N_COMP, _U_COMP)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2,
                   cam: wp.vec3, fwd: wp.vec3, rgt: wp.vec3, upv: wp.vec3,
                   wpos: wp.vec3, hull: float,
                   fpos: wp.vec3, flame: float,
                   cpos: wp.vec3,
                   shell_r: float, comp_r: float,
                   grow_frac: float, thrust_frac: float, res_frac: float):
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
            col = col + wp.vec3(0.14, 0.17, 0.30) * (0.35 * wp.exp(-(d_q * d_q) / 0.00016))
        # the hover shell and the companion's shell: working spheres, brighter
        d_sh = wp.abs(dc - shell_r)
        col = col + wp.vec3(0.30, 0.65, 0.55) * (0.45 * wp.exp(-(d_sh * d_sh) / 0.00022))
        d_cs = wp.abs(dc - comp_r)
        col = col + wp.vec3(0.45, 0.55, 0.40) * (0.25 * wp.exp(-(d_cs * d_cs) / 0.00018))

    # ---- the companion mote: omega = 1 on its own plane, in step forever ----
    cp = cpos - cam
    tcc = wp.dot(cp, rd)
    if tcc > 0.0:
        perp2 = wp.dot(cp, cp) - tcc * tcc
        col = col + wp.vec3(0.75, 0.85, 0.65) * (0.75 * wp.exp(-perp2 / 0.0005))

    # ---- the drive flame (climb + hover) ----
    if flame > 0.0:
        fp = fpos - cam
        tcf = wp.dot(fp, rd)
        if tcf > 0.0:
            perp2 = wp.dot(fp, fp) - tcf * tcf
            col = col + wp.vec3(1.00, 0.60, 0.20) * (1.5 * flame * wp.exp(-perp2 / 0.004))

    # ---- the body: wisp core + hull bubble growing around it ----
    wpr = wpos - cam
    tcw = wp.dot(wpr, rd)
    if tcw > 0.0:
        perp2 = wp.dot(wpr, wpr) - tcw * tcw
        col = col + wp.vec3(0.80, 0.95, 1.00) * (1.9 * wp.exp(-perp2 / 0.0011))
        if hull > 0.01:
            dpw = wp.sqrt(perp2)
            d_h = wp.abs(dpw - 0.075)
            col = col + wp.vec3(0.55, 0.90, 0.75) * \
                (1.0 * hull * wp.exp(-(d_h * d_h) / 0.00016))

    # ---- the ledgers: growth / thrust (with max line) / reserve ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * grow_frac:
        col = col + wp.vec3(0.45, 0.95, 0.55) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * thrust_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.48 and x < 1.62 and wp.abs(y - (-1.05 + 2.0)) < 0.007:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 1.1      # thrust max: c^2/L
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * res_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _smooth(a: float) -> float:
    a = max(0.0, min(1.0, a))
    return a * a * (3.0 - 2.0 * a)


def _state(tau: float):
    """(rho, phi, hull, flame, thrust, reserve) — same exact stage-2 arc."""
    e_climb = climb_energy(_RHO_LOW, _RHO_SHELL)
    budget = e_climb / 0.45
    if tau < _T_GROW:
        s = tau / _T_GROW
        return _RHO_LOW, _PHI0, _smooth(s), 0.0, 0.0, 0.15 + 0.85 * _smooth(s)
    if tau < _T_CLIMB:
        s = _smooth((tau - _T_GROW) / (_T_CLIMB - _T_GROW))
        rho = _RHO_LOW + (_RHO_SHELL - _RHO_LOW) * s
        spent = climb_energy(_RHO_LOW, rho) / budget
        return rho, _PHI0, 1.0, 1.0, min(1.0, hover_acceleration(rho) + 0.25), 1.0 - spent
    if tau < _T_HOVER:
        s = (tau - _T_CLIMB) / (_T_HOVER - _T_CLIMB)
        res = (1.0 - climb_energy(_RHO_LOW, _RHO_SHELL) / budget) - 0.17 * s
        return _RHO_SHELL, _PHI0, 1.0, 0.75, hover_acceleration(_RHO_SHELL), res
    s = tau - _T_HOVER
    res = (1.0 - climb_energy(_RHO_LOW, _RHO_SHELL) / budget) - 0.17
    return _RHO_SHELL, _PHI0 + s * 1.0, 1.0, 0.0, 0.0, res


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    rho, phi, hull, flame, thrust, res = _state(tau)

    rr = disk_radius(rho) * _R_BUBBLE
    wpos = rr * (math.cos(phi) * _U_BODY + math.sin(phi) * _V_BODY)
    nrm = float(np.linalg.norm(wpos))
    fdir = -wpos / nrm if nrm > 1e-6 else np.zeros(3)
    fpos = wpos + 0.10 * fdir

    # the companion: free-orbiting its own plane the whole cycle, omega = 1
    phi_c = _PHI0 - 0.9 + 1.0 * tau
    rc = disk_radius(_RHO_COMPANION) * _R_BUBBLE
    cpos = rc * (math.cos(phi_c) * _U_COMP + math.sin(phi_c) * _V_COMP)

    az = 2.0 * math.pi * (t / _T_CYCLE) + 1.2
    cam = np.array([3.4 * math.cos(az), 1.30, 3.4 * math.sin(az)])
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
                      float(hull),
                      wp.vec3(float(fpos[0]), float(fpos[1]), float(fpos[2])),
                      float(flame),
                      wp.vec3(float(cpos[0]), float(cpos[1]), float(cpos[2])),
                      float(disk_radius(_RHO_SHELL) * _R_BUBBLE),
                      float(disk_radius(_RHO_COMPANION) * _R_BUBBLE),
                      float(hull), float(thrust), float(max(res, 0.02))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_body_3d",
    description="stage 2 in 3D: the body grows inside the Poincare ball — hull "
                "bubble closing around the core low where mass is cheap, the "
                "drive lifting it up the cosh cliff to the hover SPHERE (a full "
                "translucent shell it holds altitude against, amber thrust "
                "pinned at a = tanh(rho) under the white c^2/L max line, bound "
                "asserted), then tipping into the L = r^2 orbit and cutting the "
                "flame (V'=0 asserted, stable V''=8, energy cosh^2 rho "
                "asserted). The companion mote free-orbits a DIFFERENT plane on "
                "a lower shell and stays in step lap for lap, because omega = 1 "
                "at every radius in every plane (asserted) — the whole ball "
                "turns in step. Camera orbits once per cycle. --frames runs one "
                "grow-climb-hover-orbit cycle.",
    renderer=_render,
)
