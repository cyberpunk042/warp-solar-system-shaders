"""The scale factor — the whole history of expansion in one exact curve.

Friedmann 1922: the universe's size obeys ``(ȧ/a)² = H₀²(Ω_m/a³ + Ω_Λ)``. For the
flat matter+Λ universe we live in, the solution is CLOSED FORM
(``engine.cosmology.scale_factor``):

    a(t) = (Ω_m/Ω_Λ)^{1/3} · sinh^{2/3}( (3/2)·√Ω_Λ·H₀·t )

— matter-era ``t^{2/3}`` early, de Sitter exponential late, one smooth curve between
(asserted to satisfy the Friedmann equation numerically; inverted at a = 1 it gives
t₀ = 13.8 Gyr, asserted with the real Planck numbers). The scene plays 32 Gyr of it:

* a field of **galaxies at fixed comoving positions** rides the stretching grid —
  nothing moves THROUGH space; space itself carries them apart (the rings are
  comoving markers, growing with the exact a(t));
* the panel below draws the exact curve with its two ghosts: gray ``t^{2/3}``
  (matter forever — the universe 1998 expected) and the **inflection dot** at
  ``a = (Ω_m/2Ω_Λ)^{1/3}`` where ä flips sign (z ≈ 0.63, asserted) — the brake
  becoming a throttle;
* the **"now" tick** sits at 13.8 Gyr — we live just past the handoff (ρ_m = ρ_Λ
  at z ≈ 0.3, asserted), where Λ has only recently taken the wheel;
* the amber ledger tracks a(t) as the live marker sweeps the curve and the galaxy
  field breathes outward in exact sync.

Watch the early frames crawl (gravity braking) and the late frames run away
(Λ driving) — the 2011 Nobel in one animation. --frames runs 32 Gyr; iMouse pans.
See ``docs/research/56-expanding-universe.md`` (Part I).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.cosmology import (
    OMEGA_L,
    OMEGA_M,
    acceleration_onset,
    age_at_scale_factor,
    age_of_universe,
    hubble_h0_per_gyr,
    scale_factor,
)
from ..scene import Scene

_T_CYCLE = 16.0
_T_START, _T_END = 0.6, 32.0     # Gyr shown
_N_GAL = 16


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, gal: wp.array(dtype=wp.vec2), n_gal: int,
                   a_now: float, a_end: float,
                   k1: float, k2: float, t_end: float,
                   t_now: float, t_age: float, t_acc: float, a_acc: float,
                   a_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the galaxy field: comoving positions x exact a(t), upper area ----
    gx = x
    gy = y - 0.30
    scale = 0.62 * a_now / a_end
    # comoving grid rings, stretched by a(t)
    r_g = wp.sqrt(gx * gx + gy * gy)
    wg = wp.max(0.004, 1.2 * px)
    for q in range(1, 4):
        rr = float(q) * 0.85 * scale
        d_r = wp.abs(r_g - rr)
        col = col + wp.vec3(0.16, 0.20, 0.32) * (0.35 * wp.exp(-(d_r * d_r) / (wg * wg)))
    # the galaxies (we are the center dot)
    wc = wp.max(0.016, 2.4 * px)
    col = col + wp.vec3(0.95, 0.90, 0.75) * (1.2 * wp.exp(-(r_g * r_g) / (wc * wc)))
    for g in range(n_gal):
        px_g = gal[g][0] * scale
        py_g = gal[g][1] * scale
        d2 = (gx - px_g) * (gx - px_g) + (gy - py_g) * (gy - py_g)
        # farther galaxies redden: crude Hubble tint
        rr2 = wp.sqrt(gal[g][0] * gal[g][0] + gal[g][1] * gal[g][1]) / 2.6
        tintg = wp.vec3(0.55 + 0.45 * rr2, 0.70 - 0.25 * rr2, 1.00 - 0.55 * rr2)
        col = col + tintg * (1.0 * wp.exp(-d2 / (wc * wc)))

    # ---- the panel: the exact a(t) with its ghosts ----
    if y < -0.72 and y > -1.30 and wp.abs(x) < 2.05:
        tx = (x + 2.0) / 4.0 * t_end                   # panel time axis, Gyr
        if tx > 0.0:
            xx = k2 * tx
            a_ex = k1 * wp.pow(wp.sinh(xx), 2.0 / 3.0)  # the exact curve
            a_mat = k1 * wp.pow(xx, 2.0 / 3.0)          # matter-forever ghost
            y0 = -1.26
            hgt = 0.50
            yc = y0 + hgt * a_ex / a_end
            ym = y0 + hgt * a_mat / a_end
            wl = wp.max(0.005, 1.5 * px)
            d_m = wp.abs(y - ym)
            col = col + wp.vec3(0.30, 0.34, 0.42) * (0.5 * wp.exp(-(d_m * d_m) / (wl * wl)))
            d_c = wp.abs(y - yc)
            col = col + wp.vec3(1.00, 0.82, 0.40) * (0.9 * wp.exp(-(d_c * d_c) / (wl * wl)))
            # the inflection: deceleration -> acceleration (a-double-dot = 0)
            d2i = (tx - t_acc) * (tx - t_acc) / (t_end * t_end) * 16.0 + \
                (y - (y0 + hgt * a_acc / a_end)) * (y - (y0 + hgt * a_acc / a_end)) * 40.0
            col = col + wp.vec3(0.55, 1.00, 0.75) * (1.2 * wp.exp(-d2i * 220.0))
            # the "now" tick at 13.8 Gyr
            if wp.abs(tx - t_age) < 0.10 * t_end / 32.0 * 2.0 and y < yc + 0.02:
                col = col + wp.vec3(0.35, 0.85, 1.00) * 0.5
            # live marker
            if wp.abs(tx - t_now) < 0.15 and wp.abs(y - yc) < 0.03:
                col = col + wp.vec3(1.00, 1.00, 0.95) * 1.3

    # ---- the ledger: a(t) ----
    if x > 1.62 and x < 1.70 and y > -0.55 and y < -0.55 + 1.55 * a_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


_GALS = None


def _galaxies():
    global _GALS
    if _GALS is None:
        pts = []
        for k in range(_N_GAL):
            ang = 2.399963 * float(k)                # golden-angle spiral
            rad = 0.55 + 2.05 * math.sqrt(float(k) / _N_GAL)
            pts.append((rad * math.cos(ang), rad * math.sin(ang) * 0.62))
        _GALS = np.asarray(pts, np.float32)
    return _GALS


def _render(width, height, time, mouse, device):
    tau_c = math.fmod(float(time), _T_CYCLE)
    s = tau_c / _T_CYCLE
    t_now = _T_START + s * (_T_END - _T_START)

    a_now = scale_factor(t_now)
    a_end = scale_factor(_T_END)
    h0 = hubble_h0_per_gyr()
    k1 = (OMEGA_M / OMEGA_L) ** (1.0 / 3.0)
    k2 = 1.5 * math.sqrt(OMEGA_L) * h0
    a_acc = acceleration_onset()
    t_acc = age_at_scale_factor(a_acc)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array(_galaxies(), dtype=wp.vec2, device=device), int(_N_GAL),
                      float(a_now), float(a_end),
                      float(k1), float(k2), float(_T_END),
                      float(t_now), float(age_of_universe()), float(t_acc), float(a_acc),
                      float(min(a_now / a_end, 1.0))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="cosmo_expansion",
    description="the exact flat-LCDM scale factor a(t) = (Om/OL)^(1/3) "
                "sinh^(2/3)(1.5 sqrt(OL) H0 t) — asserted to satisfy the Friedmann "
                "equation, t0 = 13.8 Gyr asserted with real Planck numbers — plays "
                "32 Gyr: galaxies at fixed comoving positions ride the stretching "
                "grid, the panel draws the exact curve past the gray "
                "matter-forever ghost with the green inflection dot where "
                "a-double-dot flips sign (z = 0.63, asserted — the 1998 "
                "signature) and the cyan now-tick at 13.8 Gyr; the amber ledger "
                "tracks a(t). Early frames crawl (gravity braking), late frames "
                "run away (Lambda driving). --frames runs the 32 Gyr.",
    renderer=_render,
)
