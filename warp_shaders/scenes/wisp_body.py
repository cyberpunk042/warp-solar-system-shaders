"""The wisp grows a body — stage 2: hover on flame, then orbit for free.

"The wisp is like the hand of the mind and the body is an engined you growth into
and hover." Stage 2 gives the wisp its body and teaches it the bubble's second
great bargain (``engine.wisp``, every law exact and test-asserted):

* **grow** — the body is assembled low, near the center, where mass is cheapest
  (static energy ``cosh ρ ≈ 1``): hull segments close around the wisp core while
  the reserve is harvested full — growth and retention before altitude;
* **climb** — the drive lifts the whole body to a hover shell at ρ_h, paying the
  exact fuel bill ``cosh ρ_h − cosh ρ₀`` from the reserve;
* **hover on flame** — holding the shell takes proper acceleration
  ``a = tanh(ρ_h) < c²/L`` (bound asserted; the amber thrust bar sits visibly
  BELOW its white max line — headroom at any altitude). But a rocket pays for
  thrust with burn time: the reserve keeps bleeding. The g it fights is real:
  released, the body would fall ``½·tanh(ρ_h)·τ²`` (equivalence principle,
  asserted against the exact geodesic);
* **orbit** — the body tips sideways and the flame CUTS: circular orbits sit at
  exactly ``L = r²`` (V′ = 0 asserted, stable V″ = 8 asserted), cost energy
  ``cosh²ρ`` (asserted — the kinetic surcharge over hover's cosh ρ), and need
  ZERO thrust forever. And the bubble's deepest signature: ``ω = 1`` EXACTLY at
  every radius (asserted) — the companion mote circling lower completes its lap
  in the same 2π. Orbiting is free hover, and the whole bubble turns in step.

Watch the ledgers tell it: green body growth, amber thrust (0.9 hovering — then
ZERO orbiting), magenta reserve (harvest, the climb bill, the hover bleed — then
FLAT). Stage 3 — navigation — comes later. --frames runs one cycle; iMouse pans.
See ``docs/research/57-the-wisp-in-the-box.md`` (Stage 2).
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


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, wx: float, wy: float, hull: float,
                   flx: float, fly: float, flame: float,
                   cx2: float, cy2: float,
                   shell_r: float,
                   grow_frac: float, thrust_frac: float, res_frac: float):
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

    # ---- the hover shell the body works at ----
    d_sh = wp.abs(r_pix - shell_r)
    col = col + wp.vec3(0.30, 0.65, 0.55) * (0.30 * wp.exp(-(d_sh * d_sh) / (wg * wg)))

    # ---- the companion mote: same omega = 1, lower shell — in step ----
    d2c = (x - cx2) * (x - cx2) + (y - cy2) * (y - cy2)
    wc2 = wp.max(0.013, 2.0 * px)
    col = col + wp.vec3(0.75, 0.85, 0.65) * (0.8 * wp.exp(-d2c / (wc2 * wc2)))

    # ---- the drive flame (during climb + hover) ----
    if flame > 0.0:
        ex = wx + flx * 0.11
        ey = wy + fly * 0.11
        d2f = (x - ex) * (x - ex) + (y - ey) * (y - ey)
        wf = wp.max(0.032, 4.2 * px)
        col = col + wp.vec3(1.00, 0.60, 0.20) * (1.6 * flame * wp.exp(-d2f / (wf * wf)))

    # ---- the body: wisp core + grown hull ring ----
    dxw = x - wx
    dyw = y - wy
    d2w = dxw * dxw + dyw * dyw
    ww = wp.max(0.018, 2.6 * px)
    col = col + wp.vec3(0.80, 0.95, 1.00) * (1.9 * wp.exp(-d2w / (ww * ww)))
    # hull: an arc of a small ring, closing as the body grows (8 dash segments)
    dw = wp.sqrt(d2w)
    d_h = wp.abs(dw - 0.085)
    ang = wp.atan2(dyw, dxw)
    if ang < 0.0:
        ang = ang + 2.0 * 3.14159265358979
    seg = ang / (2.0 * 3.14159265358979)
    if seg < hull and wp.sin(ang * 8.0) > -0.25:
        wh = wp.max(0.006, 1.5 * px)
        col = col + wp.vec3(0.55, 0.90, 0.75) * (1.0 * wp.exp(-(d_h * d_h) / (wh * wh)))

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
    """(rho, phi, hull, flame, thrust, reserve) over one cycle."""
    e_climb = climb_energy(_RHO_LOW, _RHO_SHELL)
    budget = e_climb / 0.45                       # climb spends 45% of the reserve
    if tau < _T_GROW:                             # grow + harvest, low and cheap
        s = tau / _T_GROW
        return _RHO_LOW, _PHI0, _smooth(s), 0.0, 0.0, 0.15 + 0.85 * _smooth(s)
    if tau < _T_CLIMB:                            # climb: pay cosh(rho2)-cosh(rho1)
        s = _smooth((tau - _T_GROW) / (_T_CLIMB - _T_GROW))
        rho = _RHO_LOW + (_RHO_SHELL - _RHO_LOW) * s
        spent = climb_energy(_RHO_LOW, rho) / budget
        return rho, _PHI0, 1.0, 1.0, min(1.0, hover_acceleration(rho) + 0.25), 1.0 - spent
    if tau < _T_HOVER:                            # hover on flame: thrust tanh(rho)
        s = (tau - _T_CLIMB) / (_T_HOVER - _T_CLIMB)
        res = (1.0 - climb_energy(_RHO_LOW, _RHO_SHELL) / budget) - 0.17 * s
        return _RHO_SHELL, _PHI0, 1.0, 0.75, hover_acceleration(_RHO_SHELL), res
    # orbit: flame cuts, omega = 1 exactly, reserve FLAT
    s = tau - _T_HOVER
    res = (1.0 - climb_energy(_RHO_LOW, _RHO_SHELL) / budget) - 0.17
    return _RHO_SHELL, _PHI0 + s * 1.0, 1.0, 0.0, 0.0, res


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    rho, phi, hull, flame, thrust, res = _state(tau)

    rd = disk_radius(rho) * _R_BUBBLE
    wx, wy = rd * math.cos(phi), rd * math.sin(phi)
    nrm = math.hypot(wx, wy)
    flx, fly = (-wx / nrm, -wy / nrm) if nrm > 1e-6 else (0.0, 0.0)

    # the companion mote: a free orbiter on a lower shell the whole cycle —
    # omega = 1 like every circular orbit; once the body joins, they turn in step
    phi_c = _PHI0 - 0.9 + 1.0 * tau
    rc = disk_radius(_RHO_COMPANION) * _R_BUBBLE
    cx2, cy2 = rc * math.cos(phi_c), rc * math.sin(phi_c)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(wx), float(wy), float(hull),
                      float(flx), float(fly), float(flame),
                      float(cx2), float(cy2),
                      float(disk_radius(_RHO_SHELL) * _R_BUBBLE),
                      float(hull), float(thrust), float(max(res, 0.02))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wisp_body",
    description="stage 2: the wisp grows a body and hovers with it — hull "
                "segments close low where mass is cheap (cosh rho ~ 1), the "
                "drive lifts the body to a hover shell paying the exact "
                "cosh(rho2)-cosh(rho1) bill, the flame holds it at a = tanh(rho) "
                "(amber thrust bar visibly BELOW the white c^2/L max line — "
                "bound asserted; the g it fights asserted via the equivalence "
                "principle), and then the body tips into a circular orbit and "
                "the flame CUTS: L = r^2 exactly (V'=0 asserted, stable V''=8), "
                "energy cosh^2(rho) (asserted), zero thrust forever — and "
                "omega = 1 at EVERY radius (asserted): the companion mote on a "
                "lower shell laps in the same 2pi. Orbiting is free hover; the "
                "whole bubble turns in step. --frames runs one "
                "grow-climb-hover-orbit cycle.",
    renderer=_render,
)
