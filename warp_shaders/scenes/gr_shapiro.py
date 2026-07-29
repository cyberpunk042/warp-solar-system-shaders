"""The late echo — Shapiro's fourth test of general relativity.

Shapiro 1964 noticed the metric predicts something nobody had looked for: a radar
pulse grazing the Sun returns LATE, because light near a mass travels with coordinate
speed ``c(r) = 1 − 2M/r < 1`` (``engine.geodesics.coordinate_light_speed``).
Integrated along an Earth-planet path at superior conjunction the round-trip excess is

    Δt = (4GM/c³)·[ln(4·r₁r₂/b²) + 1]

(``shapiro_roundtrip_excess``) — for Earth-Mars at the solar limb, ≈ 250 μs
(asserted): measured by Viking landers to 0.1% and by Cassini's 2002 radio link to
0.001% — still the tightest solar-system test of GR. The scene plays a conjunction:

* the planet slides behind the Sun; the radar path (drawn bowing toward the Sun by
  the exact deflection scale ``4M/b``) sweeps through the corona;
* the **pulse visibly slows** as it crosses the well — its speed IS ``1 − 2M/r``
  (exaggerated M for the eye; the ledger numbers are real) — watch it wade through
  the corona and sprint in the clear;
* the light-curve panel below draws the exact delay ``Δt(b)`` of the whole
  conjunction — the logarithmic spike as the impact parameter b sweeps through the
  solar limb — with the live marker riding it;
* the amber ledger tracks the current excess against the grazing maximum (asserted
  monotone in b in the suite).

Radio astronomy runs on this delay today: pulsar timing arrays fit the Shapiro term
of every companion, and it weighs neutron stars. --frames runs one conjunction;
iMouse pans. See ``docs/research/55-classic-tests.md`` (Part II).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.geodesics import (
    AU_KM,
    C_KM_S,
    M_SUN_KM,
    R_SUN_KM,
    coordinate_light_speed,
    shapiro_roundtrip_excess,
)
from ..scene import Scene

_T_CYCLE = 16.0
_R_MARS = 1.524 * AU_KM
_EARTH = (-1.85, 0.0)
_PLANET_X = 1.85
_R_VIS = 0.135          # the Sun's visual radius = the solar limb
_M_VIS = 0.028          # exaggerated well for the visible pulse slow-down
_T_ECHO = 2.0           # one radar round trip per _T_ECHO seconds
_M_PULSE = 0.058        # deeper well for the pulse pacing (the visible wading)
_N_PATH = 192


def _delay_us(b_scene: float) -> float:
    """Map a scene impact parameter to the REAL Earth-Mars delay in microseconds:
    grazing the visual limb = grazing the real solar limb."""
    b = max(b_scene, _R_VIS) / _R_VIS * R_SUN_KM
    return shapiro_roundtrip_excess(AU_KM, _R_MARS, b, M_SUN_KM) / C_KM_S * 1.0e6


_DELAY_MAX = _delay_us(_R_VIS)
_DELAY_MIN = _delay_us(1.85 * 1.05 / math.hypot(3.7, 1.05))   # the far wing (s = 0)
_D0_KM = _DELAY_MIN * C_KM_S / 1.0e6
_DSPAN_KM = (_DELAY_MAX - _DELAY_MIN) * C_KM_S / 1.0e6


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, py: float, cx: float, cy: float,
                   pux: float, puy: float, pulse_on: float,
                   phase: float, delay_frac: float, r_vis: float,
                   d0_km: float, dspan_km: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the Sun: disk + corona (the well the pulse wades through) ----
    r_sun = wp.sqrt(x * x + y * y)
    col = col + wp.vec3(1.00, 0.78, 0.35) * (1.6 * wp.exp(-(r_sun * r_sun) / (r_vis * r_vis)))
    col = col + wp.vec3(0.85, 0.55, 0.20) * (0.24 * wp.exp(-(r_sun - r_vis) / 0.20) *
                                             wp.where(r_sun > r_vis, 1.0, 0.0))

    # ---- Earth (blue, left) and the planet (rust, right, sliding) ----
    d2e = (x + 1.85) * (x + 1.85) + y * y
    we = wp.max(0.030, 3.5 * px)
    col = col + wp.vec3(0.30, 0.55, 1.00) * (1.5 * wp.exp(-d2e / (we * we)))
    d2p = (x - 1.85) * (x - 1.85) + (y - py) * (y - py)
    wpm = wp.max(0.024, 3.0 * px)
    col = col + wp.vec3(1.00, 0.45, 0.25) * (1.3 * wp.exp(-d2p / (wpm * wpm)))

    # ---- the radar path: quadratic bezier bowing toward the Sun ----
    # sample the curve; keep the min distance
    dmin = float(1.0e9)
    ax = -1.85
    ay = 0.0
    bx2 = 1.85
    prevx = float(ax)
    prevy = float(ay)
    for s in range(1, 25):
        t = float(s) / 24.0
        omt = 1.0 - t
        qx = omt * omt * ax + 2.0 * t * omt * cx + t * t * bx2
        qy = omt * omt * ay + 2.0 * t * omt * cy + t * t * py
        # distance to the segment (prev -> q)
        ex = qx - prevx
        ey = qy - prevy
        rxp = x - prevx
        ryp = y - prevy
        h = wp.clamp((rxp * ex + ryp * ey) / (ex * ex + ey * ey + 1.0e-12), 0.0, 1.0)
        ddx = rxp - ex * h
        ddy = ryp - ey * h
        d = wp.sqrt(ddx * ddx + ddy * ddy)
        dmin = wp.min(dmin, d)
        prevx = qx
        prevy = qy
    wl = wp.max(0.004, 1.2 * px)
    col = col + wp.vec3(0.30, 0.75, 0.75) * (0.5 * pulse_on * wp.exp(-(dmin * dmin) / (wl * wl)))

    # ---- the pulse: a bright bead wading through the well ----
    d2u = (x - pux) * (x - pux) + (y - puy) * (y - puy)
    wu = wp.max(0.016, 2.4 * px)
    col = col + wp.vec3(0.55, 1.00, 0.95) * (pulse_on * 2.2 * wp.exp(-d2u / (wu * wu)))

    # ---- the delay light-curve panel: the exact logarithmic spike ----
    # panel spans x in [-1.7, 1.7], y in [-1.28, -0.62]; curve from the REAL formula
    if y < -0.52 and wp.abs(x) < 1.78:
        sx = (x + 1.7) / 3.4                      # conjunction phase 0..1
        py_s = 1.05 * wp.cos(3.14159265358979 * sx)
        b_s = 1.85 * wp.abs(py_s) / wp.sqrt(3.7 * 3.7 + py_s * py_s)
        b_km = wp.max(b_s, 0.135) / 0.135 * 6.957e5
        dl = 4.0 * 1.4766250385 * (wp.log(4.0 * 1.495978707e8 * 2.2799e8 /
                                          (b_km * b_km)) + 1.0)
        ycur = -1.24 + 0.58 * wp.clamp((dl - d0_km) / dspan_km, 0.02, 1.0)
        d_c = wp.abs(y - ycur)
        wc2 = wp.max(0.006, 1.6 * px)
        col = col + wp.vec3(1.00, 0.72, 0.25) * (0.85 * wp.exp(-(d_c * d_c) / (wc2 * wc2)))
        # live marker riding the curve
        d_m = wp.sqrt((sx - phase) * (sx - phase) * 3.4 * 3.4 + (y - ycur) * (y - ycur))
        if wp.abs(sx - phase) < 0.02:
            col = col + wp.vec3(0.55, 1.00, 0.95) * (1.4 * wp.exp(-(d_m * d_m) / (wu * wu)))

    # ---- the ledger: current excess vs the grazing maximum ----
    if x > 1.55 and x < 1.63 and y > -0.35 and y < -0.35 + 1.05 * delay_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.1

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.32 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _pulse_state(time: float, py: float, cx: float, cy: float):
    """March the pulse along the bezier at local speed c(r) = 1 − 2M/r (host-side,
    exaggerated M): position + gating for the current echo."""
    lam = np.linspace(0.0, 1.0, _N_PATH)
    omt = 1.0 - lam
    qx = omt * omt * _EARTH[0] + 2.0 * lam * omt * cx + lam * lam * _PLANET_X
    qy = omt * omt * _EARTH[1] + 2.0 * lam * omt * cy + lam * lam * py
    seg = np.hypot(np.diff(qx), np.diff(qy))
    r = np.hypot(0.5 * (qx[1:] + qx[:-1]), 0.5 * (qy[1:] + qy[:-1]))
    c_loc = np.maximum([coordinate_light_speed(float(ri), _M_PULSE) for ri in r], 0.05)
    dt = seg / c_loc
    t_cum = np.concatenate([[0.0], np.cumsum(dt)])
    t_tot = t_cum[-1]

    p = math.fmod(time, _T_ECHO) / _T_ECHO          # one round trip per echo
    if p < 0.5:
        tt = (p / 0.5) * t_tot                       # outbound
    else:
        tt = (1.0 - (p - 0.5) / 0.5) * t_tot         # the echo comes home
    k = int(np.searchsorted(t_cum, tt))
    k = min(max(k, 0), _N_PATH - 1)
    return float(qx[k]), float(qy[k])


def _render(width, height, time, mouse, device):
    tau_c = math.fmod(float(time), _T_CYCLE)
    s = tau_c / _T_CYCLE
    py = 1.05 * math.cos(math.pi * s)                # the conjunction sweep

    # impact parameter of the Earth->planet chord (distance from the Sun)
    ex, ey = _EARTH
    dx, dy = _PLANET_X - ex, py - ey
    b_scene = abs(ex * dy - dx * ey) / math.hypot(dx, dy)

    # the bow: pull the midpoint toward the Sun by the 4M/b scale, never inside
    # the limb; during the occultation window the link blacks out (real radar
    # blackouts at superior conjunction) and the path fades
    side = 1.0 if py >= 0.0 else -1.0
    pull = min(2.0 * _M_VIS / max(b_scene, 0.08), 0.32)
    m_y = side * max(abs(0.5 * py) - pull, 1.25 * _R_VIS)
    cx, cy = 0.0, 2.0 * m_y - 0.5 * py
    occult = min(max((b_scene / _R_VIS - 1.02) / 0.25, 0.10), 1.0)

    pux, puy = _pulse_state(float(time), py, cx, cy)
    d_now = _delay_us(b_scene)
    delay_frac = min(max((d_now - _DELAY_MIN) / (_DELAY_MAX - _DELAY_MIN), 0.02), 1.0)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(py), float(cx), float(cy),
                      float(pux), float(puy), float(occult),
                      float(s), float(delay_frac), float(_R_VIS),
                      float(_D0_KM), float(_DSPAN_KM)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gr_shapiro",
    description="the fourth test — a radar pulse to Mars sweeps through superior "
                "conjunction on a path bowing toward the Sun, visibly slowing in "
                "the well because light's coordinate speed is 1 - 2M/r; the panel "
                "below draws the EXACT round-trip excess 4M[ln(4 r1 r2/b^2)+1] — "
                "the logarithmic spike as b sweeps the solar limb, ~250 us for "
                "Earth-Mars (asserted; Viking measured it to 0.1%, Cassini to "
                "0.001%) — with a live marker riding the curve and the amber "
                "ledger tracking the current excess. --frames runs one "
                "conjunction.",
    renderer=_render,
)
