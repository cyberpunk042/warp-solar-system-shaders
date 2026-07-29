"""The clock lattice — gravitational time dilation, from Pound-Rebka to GPS.

The oldest prediction (Einstein 1907, from the equivalence principle alone, eight
years before the field equations): clocks deeper in a potential run slower. A clock
standing at r ticks at ``√(1−2M/r)``; a clock on a CIRCULAR ORBIT at r ticks at
``√(1−3M/r)`` — altitude blueshift and orbital time dilation folded into one exact
term (``engine.geodesics.clock_rate_static`` / ``clock_rate_orbit``). The scene runs
four real clocks around one planet (rates exact; hand divergence amplified for the
eye — the numbers in the suite are real):

* the **ground clock** and the **tower clock** above it — the Pound-Rebka pair:
  the upper clock runs measurably fast (gh/c² ≈ 2.46×10⁻¹⁵ over Harvard's 22.5 m,
  asserted — resolved in 1959 with the Mössbauer effect);
* the **low orbiter** (ISS) — BELOW the crossover, speed wins: its clock runs SLOW
  (astronauts really age less — by ~0.01 s per six months); tinted red;
* the **high orbiter** (GPS) — ABOVE the crossover, altitude wins: its clock runs
  FAST, +38.5 μs/day for the real constellation (asserted; uncorrected, fixes would
  drift ~11 km/day — relativity as an engineering requirement); tinted blue;
* the **dashed ring** is the break-even orbit ``r = 3R/2`` — where ``1−3M/r``
  equals the ground's ``1−2M/R`` EXACTLY, independent of the planet's mass
  (asserted): pure geometry deciding whether a clock gains or loses;
* the amber/magenta ledger bars accumulate the GPS gain (up) and the ISS loss
  (down) against the ground clock, live.

Every GPS fix you have ever taken silently applied this scene. --frames runs one
lattice cycle; iMouse pans. See ``docs/research/55-classic-tests.md`` (Part III).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.geodesics import clock_crossover_radius, clock_rate_orbit, clock_rate_static
from ..scene import Scene

_T_CYCLE = 16.0
_CEN = (0.42, -0.58)         # planet center (low, right of middle)
_R = 0.52                    # planet radius
_M_VIS = 0.0156              # exaggerated mass: visible rate differences
_R_ISS = 1.25 * _R
_R_GPS = 2.15 * _R
_TOWER_PHI = math.radians(118.0)   # the Pound-Rebka tower: up-left on the limb
_GAIN = 22.0                 # hand-divergence amplification (rates stay exact)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, cenx: float, ceny: float, r_pl: float,
                   r_iss: float, r_gps: float, r_cross: float,
                   pos: wp.array(dtype=wp.vec2), hand: wp.array(dtype=float),
                   tint: wp.array(dtype=wp.vec3), n_clk: int,
                   gps_frac: float, iss_frac: float,
                   tbx: float, tby: float, ttx: float, tty: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    dxp = x - cenx
    dyp = y - ceny
    r_c = wp.sqrt(dxp * dxp + dyp * dyp)
    phi = wp.atan2(dyp, dxp)

    # ---- the planet: body + limb ----
    if r_c < r_pl:
        depth = 1.0 - r_c / r_pl
        col = col + wp.vec3(0.10, 0.16, 0.24) * (0.5 + 0.5 * depth)
    d_l = wp.abs(r_c - r_pl)
    wl = wp.max(0.006, 1.8 * px)
    col = col + wp.vec3(0.35, 0.60, 0.85) * (0.7 * wp.exp(-(d_l * d_l) / (wl * wl)))

    # ---- the orbits: faint circles for ISS and GPS ----
    wg = wp.max(0.004, 1.2 * px)
    d_i = wp.abs(r_c - r_iss)
    col = col + wp.vec3(0.55, 0.30, 0.30) * (0.30 * wp.exp(-(d_i * d_i) / (wg * wg)))
    d_g = wp.abs(r_c - r_gps)
    col = col + wp.vec3(0.30, 0.40, 0.65) * (0.30 * wp.exp(-(d_g * d_g) / (wg * wg)))

    # ---- the break-even ring r = 3R/2: dashed, mass-independent ----
    d_x = wp.abs(r_c - r_cross)
    dash = wp.sin(phi * 22.0)
    if dash > 0.0:
        col = col + wp.vec3(0.55, 1.00, 0.75) * (0.55 * wp.exp(-(d_x * d_x) / (wg * wg)))

    # ---- the Pound-Rebka tower: a radial mast on the limb ----
    ex = ttx - tbx
    ey = tty - tby
    rxp = x - tbx
    ryp = y - tby
    ht = wp.clamp((rxp * ex + ryp * ey) / (ex * ex + ey * ey + 1.0e-12), 0.0, 1.0)
    tdx = rxp - ex * ht
    tdy = ryp - ey * ht
    d_tw = wp.sqrt(tdx * tdx + tdy * tdy)
    wt = wp.max(0.006, 1.6 * px)
    col = col + wp.vec3(0.40, 0.48, 0.58) * (0.7 * wp.exp(-(d_tw * d_tw) / (wt * wt)))

    # ---- the clocks: dial ring + hand + rate tint ----
    for c in range(n_clk):
        ddx = x - pos[c][0]
        ddy = y - pos[c][1]
        d2 = ddx * ddx + ddy * ddy
        d = wp.sqrt(d2)
        rad = 0.085
        # face
        if d < rad:
            col = col + tint[c] * 0.20
        # rim
        d_r = wp.abs(d - rad)
        wr = wp.max(0.005, 1.4 * px)
        col = col + tint[c] * (1.1 * wp.exp(-(d_r * d_r) / (wr * wr)))
        # hand: segment from center toward (cos a, sin a)
        hx = wp.cos(hand[c])
        hy = wp.sin(hand[c])
        t = wp.clamp(ddx * hx + ddy * hy, 0.0, rad * 0.78)
        sx = ddx - hx * t
        sy = ddy - hy * t
        d_h = wp.sqrt(sx * sx + sy * sy)
        wh = wp.max(0.005, 1.3 * px)
        col = col + wp.vec3(0.95, 0.95, 0.90) * (1.3 * wp.exp(-(d_h * d_h) / (wh * wh)))
        # 12 o'clock reference tick
        ty = pos[c][1] + rad * 0.85
        d_t = wp.sqrt(ddx * ddx + (y - ty) * (y - ty))
        col = col + wp.vec3(0.95, 0.95, 0.90) * (0.5 * wp.exp(-(d_t * d_t) / (wh * wh)))

    # ---- the drift ledger: GPS gains (amber, up), ISS loses (magenta, down) ----
    if x > 1.55 and x < 1.63 and y > 0.02 and y < 0.02 + 1.05 * gps_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.1
    if x > 1.55 and x < 1.63 and y < -0.02 and y > -0.02 - 1.05 * iss_frac:
        col = col + wp.vec3(1.00, 0.35, 0.60) * 1.1
    if x > 1.51 and x < 1.67 and wp.abs(y) < 0.006:
        col = col + wp.vec3(0.60, 0.65, 0.70) * 0.9

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.32 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _rates():
    ground = clock_rate_static(_R, _M_VIS)
    tower = clock_rate_static(_R + 0.40, _M_VIS)
    iss = clock_rate_orbit(_R_ISS, _M_VIS)
    gps = clock_rate_orbit(_R_GPS, _M_VIS)
    return ground, tower, iss, gps


def _render(width, height, time, mouse, device):
    t = float(time)
    ground, tower, iss, gps = _rates()
    cx, cy = _CEN

    # orbiters: the low one laps the high one (Kepler, rounded to whole
    # revolutions per cycle so the loop closes seamlessly)
    th_iss = 1.35 + 2.0 * math.pi * 2.0 * t / _T_CYCLE
    th_gps = 1.90 + 2.0 * math.pi * 1.0 * t / _T_CYCLE
    p_iss = (cx + _R_ISS * math.cos(th_iss), cy + _R_ISS * math.sin(th_iss))
    p_gps = (cx + _R_GPS * math.cos(th_gps), cy + _R_GPS * math.sin(th_gps))
    ux, uy = math.cos(_TOWER_PHI), math.sin(_TOWER_PHI)
    tb = (cx + _R * ux, cy + _R * uy)                 # mast base on the surface
    tt = (cx + (_R + 0.46) * ux, cy + (_R + 0.46) * uy)
    p_ground = (cx + (_R + 0.11) * ux, cy + (_R + 0.11) * uy)
    p_tower = (cx + (_R + 0.40) * ux, cy + (_R + 0.40) * uy)

    # hands: exact rates, divergence amplified for the eye
    base = -2.0 * math.pi * (4.0 / _T_CYCLE) * t + 0.5 * math.pi
    def ang(rate):
        return base * (1.0 + _GAIN * (rate / ground - 1.0))
    hands = [ang(ground), ang(tower), ang(iss), ang(gps)]
    poss = [p_ground, p_tower, p_iss, p_gps]
    tints = [(0.60, 0.75, 0.70), (0.55, 0.80, 0.95),
             (1.00, 0.45, 0.45), (0.45, 0.65, 1.00)]

    # accumulated drift vs ground over the cycle
    tau_c = math.fmod(t, _T_CYCLE)
    gps_frac = min((gps / ground - 1.0) * _GAIN * tau_c / _T_CYCLE * 2.4, 1.0)
    iss_frac = min((1.0 - iss / ground) * _GAIN * tau_c / _T_CYCLE * 2.4, 1.0)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(cx), float(cy), float(_R),
                      float(_R_ISS), float(_R_GPS),
                      float(clock_crossover_radius(_R)),
                      wp.array(np.asarray(poss, np.float32), dtype=wp.vec2, device=device),
                      wp.array(np.asarray(hands, np.float32), dtype=float, device=device),
                      wp.array(np.asarray(tints, np.float32), dtype=wp.vec3, device=device),
                      int(len(poss)),
                      float(gps_frac), float(iss_frac),
                      float(tb[0]), float(tb[1]), float(tt[0]), float(tt[1])],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gr_clocks",
    description="gravitational time dilation as a working clock lattice — the "
                "Pound-Rebka tower pair (upper clock fast: gh/c^2 = 2.46e-15 over "
                "22.5 m, asserted), a red-tinted low orbiter BELOW the dashed "
                "break-even ring r = 3R/2 whose clock runs SLOW (speed wins: ISS "
                "astronauts age less), and a blue-tinted high orbiter ABOVE it "
                "running FAST (+38.5 us/day for real GPS, asserted) — hands tick "
                "at the exact rates sqrt(1-2M/r) static / sqrt(1-3M/r) orbiting "
                "(divergence amplified for the eye), while the ledger accumulates "
                "the GPS gain (amber, up) and ISS loss (magenta, down). The "
                "crossover is mass-independent: pure geometry. --frames runs one "
                "cycle.",
    renderer=_render,
)
