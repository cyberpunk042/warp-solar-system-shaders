"""The rosette — Mercury's perihelion, integrated exactly.

Le Verrier 1859: after every planet tugs on every other, Mercury's perihelion still
creeps an unexplained 43″ per century. Einstein, November 1915: the Schwarzschild
geodesic explains it to the arcsecond — "for a few days, I was beside myself with
joyous excitement." The scene integrates that EXACT geodesic live — RK4 on the Binet
equation (``engine.geodesics.integrate_orbit``)

    d²u/dφ² + u = M/L² + 3Mu²,   u = 1/r

whose ``3Mu²`` term IS general relativity — drop it and the ghost ellipse returns:

* the **cyan trail** is the exact integrated orbit, run in a strong field (a = 26M,
  e = 0.5) so each orbit visibly advances its perihelion — the ellipse becomes a
  rosette, paced physically (dτ/dφ = r²/L: the body sprints through perihelion,
  hangs at aphelion);
* the **gray ghost** is Newton's closed ellipse — same a, same e, no 3Mu² — the
  orbit that never precesses;
* **amber dots** mark each perihelion passage: the apsis line walks around the sun;
* the amber ledger fills with the accumulated advance; the cyan tick on it is
  Einstein's first-order formula ``6πM/(a(1−e²))`` — in this strong field the exact
  orbit overshoots it (asserted in the suite; for Mercury's weak field they agree to
  0.4%, and the engine's ``mercury_precession_arcsec_century()`` returns 42.98″,
  asserted).

The photon sphere (3M, faint) and ISCO (6M, dashed) rim the central mass — the
strong-field furniture Mercury never sees but this orbit skims. --frames runs the
full five-orbit rosette; iMouse pans. See ``docs/research/55-classic-tests.md``
(Part I).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.geodesics import (
    integrate_orbit,
    isco_radius,
    measured_precession,
    photon_sphere,
    precession_per_orbit,
)
from ..scene import Scene

_T_CYCLE = 16.0
_A, _E, _M = 26.0, 0.5, 1.0
_N_ORBITS = 5
_N_STEPS = 24000


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, r_tab: wp.array(dtype=float), n_tab: int,
                   dphi: float, head_idx: float, n_wraps: int,
                   hx: float, hy: float, peri_flare: float,
                   a_s: float, ecc: float,
                   peris: wp.array(dtype=wp.vec2), n_peri: int,
                   r_ph: float, r_isco: float,
                   adv_frac: float, tick_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # orbit frame shifted right so Newton's aphelion (left of focus) balances
    xo = x + 0.28
    r_pix = wp.sqrt(xo * xo + y * y)
    phi = wp.atan2(y, xo)
    if phi < 0.0:
        phi = phi + 2.0 * 3.14159265358979

    # ---- Newton's ghost: the closed ellipse the 3Mu^2 term breaks ----
    r_ell = a_s * (1.0 - ecc * ecc) / (1.0 + ecc * wp.cos(phi))
    d_g = wp.abs(r_pix - r_ell)
    wg = wp.max(0.005, 1.4 * px)
    col = col + wp.vec3(0.16, 0.19, 0.28) * (0.55 * wp.exp(-(d_g * d_g) / (wg * wg)))

    # ---- the exact geodesic trail: r(phi) looked up per wrap ----
    wo = wp.max(0.006, 1.6 * px)
    for k in range(n_wraps):
        phi_tot = phi + 2.0 * 3.14159265358979 * float(k)
        idx = phi_tot / dphi
        if idx < head_idx and int(idx) < n_tab - 1:
            i0 = int(idx)
            f = idx - float(i0)
            r_orb = r_tab[i0] * (1.0 - f) + r_tab[i0 + 1] * f
            d_o = wp.abs(r_pix - r_orb)
            # recency fade: the trail cools behind the head
            behind = (head_idx - idx) * dphi
            fade = 0.30 + 0.70 * wp.exp(-behind / 9.0)
            col = col + wp.vec3(0.35, 0.85, 1.00) * \
                (0.85 * fade * wp.exp(-(d_o * d_o) / (wo * wo)))

    # ---- perihelion markers: the walking apsis line ----
    for p in range(n_peri):
        d2p = (xo - peris[p][0]) * (xo - peris[p][0]) + (y - peris[p][1]) * (y - peris[p][1])
        wpk = wp.max(0.014, 2.2 * px)
        col = col + wp.vec3(1.00, 0.72, 0.25) * (1.1 * wp.exp(-d2p / (wpk * wpk)))

    # ---- the central mass + strong-field furniture ----
    wc = wp.max(0.020, 3.0 * px)
    col = col + wp.vec3(0.95, 0.85, 0.65) * (1.5 * wp.exp(-(r_pix * r_pix) / (wc * wc)))
    d_ph = wp.abs(r_pix - r_ph)
    col = col + wp.vec3(0.85, 0.60, 0.30) * (0.30 * wp.exp(-(d_ph * d_ph) / (wg * wg)))
    d_is = wp.abs(r_pix - r_isco)
    dash = wp.sin(phi * 14.0)
    if dash > 0.0:
        col = col + wp.vec3(0.30, 0.45, 0.70) * (0.30 * wp.exp(-(d_is * d_is) / (wg * wg)))

    # ---- the body ----
    d2b = (xo - hx) * (xo - hx) + (y - hy) * (y - hy)
    wb = wp.max(0.018, 2.6 * px)
    col = col + wp.vec3(0.95, 0.95, 0.90) * (1.7 * wp.exp(-d2b / (wb * wb)))
    col = col + wp.vec3(1.00, 0.55, 0.20) * (0.8 * peri_flare * wp.exp(-d2b / (wb * wb * 6.0)))

    # ---- the ledger: accumulated advance (amber) vs the first-order tick (cyan) ----
    if x > 1.52 and x < 1.60 and y > -1.05 and y < -1.05 + 2.0 * adv_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.48 and x < 1.64 and wp.abs(y - (-1.05 + 2.0 * tick_frac)) < 0.008:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.2

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


_CACHE = None


def _traj():
    global _CACHE
    if _CACHE is None:
        adv = measured_precession(_A, _E, _M)
        period = 2.0 * math.pi + adv
        phi_max = _N_ORBITS * period
        phis, us = integrate_orbit(_A, _E, _M, _N_STEPS, phi_max)
        scale = 1.16 / (_A * (1.0 + _E))
        r = np.asarray([1.0 / u for u in us], np.float32) * scale
        dphi = phis[1] - phis[0]
        # physical pacing: dtau/dphi = r^2/L
        ell = math.sqrt(_M * _A * (1.0 - _E * _E))
        rr = np.asarray([1.0 / u for u in us], np.float64)
        tau = np.cumsum(rr * rr / ell) * dphi
        tau = tau / tau[-1]
        # perihelion passages: local maxima of u
        peri = []
        for k in range(2, len(us) - 1):
            if us[k] >= us[k - 1] and us[k] >= us[k + 1]:
                peri.append((phis[k], (1.0 / us[k]) * scale))
        _CACHE = (r, dphi, tau, peri, adv, precession_per_orbit(_A, _E, _M), scale)
    return _CACHE


def _render(width, height, time, mouse, device):
    r_tab, dphi, tau, peri, adv, adv_formula, scale = _traj()
    tau_c = math.fmod(float(time), _T_CYCLE)
    s = tau_c / _T_CYCLE
    head_idx = float(np.searchsorted(tau, s))
    head_idx = min(head_idx, float(len(r_tab) - 1))
    hi = int(head_idx)
    phi_h = hi * dphi
    r_h = float(r_tab[hi])
    hx, hy = r_h * math.cos(phi_h), r_h * math.sin(phi_h)

    # perihelion flare: near a u-maximum the body is deepest and fastest
    r_p = _A * (1.0 - _E) * scale
    peri_flare = math.exp(-((r_h - r_p) / (0.08 * r_p)) ** 2)

    # markers only for perihelia already passed
    pts = [(rp * math.cos(ph), rp * math.sin(ph)) for (ph, rp) in peri if ph <= phi_h]
    if not pts:
        pts = [(r_p, 0.0)]
    pts = pts[:8]

    total_adv = _N_ORBITS * adv
    adv_frac = min(phi_h / (2.0 * math.pi + adv) * adv / total_adv, 1.0)
    tick_frac = adv_formula / adv                      # first-order underestimate

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array(r_tab, dtype=float, device=device), int(len(r_tab)),
                      float(dphi), float(head_idx), int(_N_ORBITS + 2),
                      float(hx), float(hy), float(peri_flare),
                      float(_A * scale), float(_E),
                      wp.array(np.asarray(pts, np.float32), dtype=wp.vec2, device=device),
                      int(len(pts)),
                      float(photon_sphere(_M) * scale), float(isco_radius(_M) * scale),
                      float(adv_frac), float(min(tick_frac, 1.0))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gr_precession",
    description="Mercury's perihelion, integrated exactly — RK4 on the Binet "
                "equation u'' + u = M/L^2 + 3Mu^2 in a strong field (a=26M, e=0.5): "
                "the cyan trail precesses into a rosette past Newton's gray closed "
                "ellipse (drop the 3Mu^2 term and it returns), amber dots mark the "
                "walking perihelion, the body sprints through perihelion on physical "
                "pacing dtau/dphi = r^2/L, and the ledger fills with the measured "
                "advance past the cyan tick of Einstein's 6piM/(a(1-e^2)) — the "
                "exact orbit overshoots the formula here (asserted; for the real "
                "Mercury the engine returns 42.98 arcsec/century). Photon sphere "
                "and ISCO rim the mass. --frames runs the five-orbit rosette.",
    renderer=_render,
)
