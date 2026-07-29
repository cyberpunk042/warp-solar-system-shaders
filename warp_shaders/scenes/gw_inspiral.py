"""Binary inspiral — the last orbits, radiating a two-armed spiral of spacetime.

Two black holes orbit their common center; the orbit is losing energy to
gravitational radiation and the entire scene runs on the exact leading-order
dictionary (``engine.gw``):

* the separation follows the exact inspiral trajectory
  ``a(t) = (256/5·m₁m₂M·(t_c−t))^{1/4}`` (``separation_of_time_left``, asserted to
  invert Peters' ``T = 5a⁴/256m₁m₂M``);
* the orbital phase accumulates with Kepler's ``Ω = √(M/a³)`` — the closed form
  ``φ ∝ (t_c−t)^{5/8}`` — so the pair visibly speeds up as it tightens;
* the radiated pattern is the **two-armed spiral** ``cos(2θ − 2φ_orb + 2Ω·r)``:
  quadrupole radiation oscillates at TWICE the orbital frequency (``f_gw = 2f_orb``,
  asserted), which is exactly why the spiral has two arms — the pattern repeats every
  half orbit;
* the wave amplitude climbs with the pitch as ``h ∝ M_c^{5/3}f^{2/3}``
  (``strain_amplitude``), falling as 1/r outward.

Each cycle plays the final stretch of the inspiral in slow motion: the orbit shrinks
by the fourth-power law (halve the separation, sixteenfold less time left — asserted),
the spiral winds tighter and brighter, the pair merges in a flash, a few damped
ringdown ripples wash outward, and the cycle re-arms. This is GW150914's last second,
drawn from the formulas that measured it. --frames runs one full inspiral; iMouse
pans. See ``docs/research/53-gravitational-waves.md`` (Part I).
"""

import math

import warp as wp

from ..engine import post
from ..engine.gw import (
    chirp_mass,
    gw_frequency,
    peters_merger_time,
    separation_of_time_left,
    strain_amplitude,
)
from ..scene import Scene

_T_CYCLE = 16.0
_T_MERGE = 13.0
_M1, _M2 = 1.0, 1.0
_A_START = 1.05                          # separation at cycle start (display units)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, sep: float, phi: float, omega: float, amp: float,
                   merged: float, ring_t: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)
    r = wp.sqrt(x * x + y * y)
    theta = wp.atan2(y, x)

    # ---- the two-armed spiral: quadrupole radiation at f_gw = 2 f_orb ----
    if merged < 0.5:
        arg = 2.0 * theta - 2.0 * phi + 2.0 * omega * r * 1.35
        wavefield = wp.cos(arg)
        envelope = amp / wp.max(r, 0.14)
        v = wavefield * envelope
        warmth = wp.min(amp * 1.4, 1.0)
        wcol = wp.vec3(0.30 + 0.55 * warmth, 0.55, 1.00 - 0.35 * warmth)
        col = col + wcol * (wp.max(v, 0.0) * 0.55)
        col = col + wp.vec3(0.10, 0.16, 0.38) * (wp.max(-v, 0.0) * 0.35)

        # ---- the pair ----
        for s in range(2):
            sgn = 1.0
            if s == 1:
                sgn = -1.0
            bx = sgn * 0.5 * sep * wp.cos(phi)
            by = sgn * 0.5 * sep * wp.sin(phi)
            d2 = (x - bx) * (x - bx) + (y - by) * (y - by)
            wb = wp.max(0.030, 3.0 * px)
            col = col + wp.vec3(0.95, 0.90, 0.80) * (1.8 * wp.exp(-d2 / (wb * wb)))
            # photon-ring hint
            col = col + wp.vec3(1.00, 0.65, 0.25) * (0.5 * wp.exp(-(wp.sqrt(d2) - 1.9 * wb) * (wp.sqrt(d2) - 1.9 * wb) / (wb * wb * 0.25)))
    else:
        # ---- merger flash + ringdown ripples washing outward ----
        col = col + wp.vec3(1.00, 0.95, 0.85) * (1.6 * wp.exp(-ring_t * 3.0) * wp.exp(-r * r * 2.2))
        ripple = wp.cos(9.0 * (r - 1.15 * ring_t)) * wp.exp(-ring_t * 1.4) * wp.exp(-r * 0.8)
        if r < 1.15 * ring_t + 0.35:
            col = col + wp.vec3(0.55, 0.45, 1.00) * (wp.max(ripple, 0.0) * 0.65)
        # the remnant
        d2 = x * x + y * y
        wb = wp.max(0.052, 4.0 * px)
        col = col + wp.vec3(0.20, 0.22, 0.30) * (0.9 * wp.exp(-d2 / (wb * wb)))
        col = col + wp.vec3(1.00, 0.60, 0.20) * (0.8 * wp.exp(-(wp.sqrt(d2) - 2.0 * wb) * (wp.sqrt(d2) - 2.0 * wb) / (wb * wb * 0.2)))

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.55, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)
    m_tot = _M1 + _M2
    mc = chirp_mass(_M1, _M2)

    # scale the exact inspiral so a(_A_START) merges at _T_MERGE
    t_full = peters_merger_time(_A_START, _M1, _M2)
    speed = t_full / _T_MERGE                     # physical seconds per scene second

    if tau < _T_MERGE:
        t_left = (_T_MERGE - tau) * speed
        a = separation_of_time_left(t_left, _M1, _M2)
        om_orb = math.sqrt(m_tot / a ** 3)                              # Kepler
        # closed-form accumulated phase: phi = -(8/5) * Omega(t_left) * t_left / speed... in scene time
        phi = -(8.0 / 5.0) * om_orb * t_left / speed
        f_now = gw_frequency(a, m_tot)
        amp = strain_amplitude(f_now, mc, 8.0)
        amp = min(amp / strain_amplitude(gw_frequency(0.16, m_tot), mc, 8.0), 1.0)
        merged, ring_t = 0.0, 0.0
        # cap the SPATIAL winding so the spiral stays resolvable (the true wavelength
        # near merger is far below pixel scale — pure moire otherwise); temporal
        # speed-up stays exact via phi
        om_disp = min(om_orb / speed, 9.0)
    else:
        a, phi = 0.0, 0.0
        om_disp, amp = 0.0, 0.0
        merged, ring_t = 1.0, tau - _T_MERGE

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a), float(phi), float(om_disp), float(amp),
                      float(merged), float(ring_t)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=6)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gw_inspiral",
    description="binary black-hole inspiral on the exact Peters trajectory "
                "a(t) = (256/5 m1 m2 M (tc-t))^{1/4} — the pair speeds up on "
                "Kepler's law while radiating a TWO-armed spiral (quadrupole: "
                "f_gw = 2 f_orb, asserted), the wave climbing in pitch and "
                "amplitude as h ~ Mc^{5/3} f^{2/3} until merger: flash, damped "
                "ringdown ripples, re-arm. GW150914's last second drawn from the "
                "formulas that measured it. --frames runs one inspiral.",
    renderer=_render,
)
