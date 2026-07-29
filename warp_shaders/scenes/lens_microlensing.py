"""Microlensing — the Paczyński light curve, drawn from the exact sum rule.

When the two images of a lensed star are too close to resolve (stellar lenses: the
image splitting is micro-arcseconds), all you see is the combined brightness — and it
is exactly

    A(u) = (u² + 2)/(u·√(u² + 4)),    u = β/θ_E

(``engine.lensing.paczynski_magnification``; asserted to equal |μ₊|+|μ₋| from the
exact image solution, with the beautiful signed sum rule μ₊+μ₋ = 1 asserted alongside,
and the anchors A(1) = 3/√5, A(∞) = 1). Paczyński 1986 proposed staring at millions of
bulge stars for this signature; OGLE and MOA have now logged tens of thousands of
events, weighed free-floating planets, and found worlds by the kink a planet adds to
this very curve.

* **top panel** — the sky: the source star slides along its track behind the lens;
  its two images (computed from the exact ``θ± = (β±√(β²+4θ_E²))/2``) slide along the
  source-lens axis, the outer one swelling, the inner counter-image brightening as it
  rises toward the ring — each drawn with brightness ∝ its exact |μ|;
* **bottom panel** — the light curve ``A(u(t))`` revealed live behind a sweep line:
  flat, the smooth achromatic rise, the peak at closest approach (u₀ = 0.25 →
  A ≈ 4.1), the symmetric fall. No color change, no repeat — the signature that
  separates lensing from every variable star.

--frames runs one transit; iMouse pans. See
``docs/research/54-gravitational-lensing.md`` (Part II).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.lensing import image_positions, magnifications, paczynski_magnification
from ..scene import Scene

_T_CYCLE = 16.0
_U0 = 0.25                                # impact parameter in Einstein radii
_N_COL = 960


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, curve: wp.array(dtype=float), n_col: int, reveal: float,
                   lx: float, ly: float, sxp: float, syp: float,
                   i1x: float, i1y: float, i1b: float, i2x: float, i2y: float, i2b: float,
                   theta_e: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    u = fx / res[0]
    v = fy / res[1]
    px = 1.0 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ======== top panel: the sky (v in [0.42, 1.0]) ========
    if v > 0.42:
        # panel coords
        sx = (u - 0.5) * 2.4
        sy = (v - 0.71) * 2.4
        spx = 2.4 * px

        # Einstein ring marker around the lens
        dl = wp.sqrt((sx - lx) * (sx - lx) + (sy - ly) * (sy - ly))
        d_ring = wp.abs(dl - theta_e)
        wr = wp.max(0.004, 1.2 * spx)
        dash = 0.5 + 0.5 * wp.sin(wp.atan2(sy - ly, sx - lx) * 20.0)
        col = col + wp.vec3(0.60, 0.35, 0.95) * (0.20 * dash * wp.exp(-(d_ring * d_ring) / (wr * wr)))

        # the lens star (foreground, dim red dwarf)
        wl = wp.max(0.014, 2.2 * spx)
        col = col + wp.vec3(0.85, 0.40, 0.22) * (0.9 * wp.exp(-dl * dl / (wl * wl)))

        # the true (unobservable) source position: ghost outline
        ds2 = (sx - sxp) * (sx - sxp) + (sy - syp) * (sy - syp)
        col = col + wp.vec3(0.30, 0.40, 0.55) * (0.30 * wp.exp(-ds2 / (wl * wl * 1.4)))

        # the two images, brightness ~ exact |mu|
        d12 = (sx - i1x) * (sx - i1x) + (sy - i1y) * (sy - i1y)
        col = col + wp.vec3(0.75, 0.90, 1.00) * (i1b * wp.exp(-d12 / (wl * wl)))
        d22 = (sx - i2x) * (sx - i2x) + (sy - i2y) * (sy - i2y)
        col = col + wp.vec3(0.55, 0.80, 1.00) * (i2b * wp.exp(-d22 / (wl * wl * 0.7)))

        # the source track, faint horizontal line at impact parameter
        if wp.abs(sy - syp) < 0.9 * spx and wp.abs(sx) < 1.15:
            col = col + wp.vec3(0.10, 0.12, 0.18)

    # ======== bottom panel: the light curve (v in [0.03, 0.38]) ========
    if v > 0.03 and v < 0.40 and u > 0.02 and u < 0.98:
        ci = int(u * float(n_col))
        if ci > n_col - 1:
            ci = n_col - 1
        if u < reveal:
            av = curve[ci]                          # 0..1 normalized magnification
            yc = 0.05 + 0.31 * av
            dc = wp.abs(v - yc)
            wc = wp.max(0.004, 1.6 * px)
            warm = wp.vec3(0.35 + 0.60 * av, 0.75 - 0.15 * av, 1.00 - 0.55 * av)
            col = col + warm * (wp.exp(-(dc * dc) / (wc * wc)) * 1.2)
            if v > 0.05 and v < yc:
                col = col + warm * (0.06 * wp.exp(-(yc - v) * 30.0))
        # baseline A = 1
        yb = 0.05 + 0.31 * 0.0
        if wp.abs(v - yb) < 0.6 * px:
            col = col + wp.vec3(0.06, 0.07, 0.11)
        # sweep line
        if wp.abs(u - reveal) < 1.2 * px:
            col = col + wp.vec3(0.95, 0.90, 0.75) * 0.55

    uvx = u - 0.5
    uvy = v - 0.5
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


_CURVE = None


def _render(width, height, time, mouse, device):
    global _CURVE
    theta_e = 0.42
    if _CURVE is None:
        cv = np.zeros(_N_COL, np.float32)
        a_max = paczynski_magnification(_U0)
        for c in range(_N_COL):
            tt = (c + 0.5) / _N_COL * _T_CYCLE
            ux = (tt / _T_CYCLE - 0.5) * 7.0
            uu = math.sqrt(ux * ux + _U0 * _U0)
            cv[c] = (paczynski_magnification(uu) - 1.0) / (a_max - 1.0)
        _CURVE = cv
    tau = math.fmod(float(time), _T_CYCLE)
    reveal = 0.02 + 0.96 * (tau / _T_CYCLE)

    # sky-panel geometry (all in Einstein-radius-scaled panel units)
    lx, ly = 0.0, 0.0
    ux = (tau / _T_CYCLE - 0.5) * 7.0            # source track in units of theta_E
    sxp = ux * theta_e
    syp = _U0 * theta_e
    beta = math.sqrt(ux * ux + _U0 * _U0)        # in theta_E units
    tp, tm = image_positions(beta, 1.0)          # in theta_E units along the axis
    mp, mm = magnifications(beta, 1.0)
    # place images along the lens->source direction
    nrm = max(beta, 1e-9)
    dxn, dyn = (ux / nrm, _U0 / nrm)
    i1x, i1y = tp * dxn * theta_e, tp * dyn * theta_e
    i2x, i2y = tm * dxn * theta_e, tm * dyn * theta_e
    i1b = min(abs(mp), 4.5) * 0.55
    i2b = min(abs(mm), 4.5) * 0.55

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array(_CURVE, dtype=float, device=device), int(_N_COL),
                      float(reveal), float(lx), float(ly), float(sxp), float(syp),
                      float(i1x), float(i1y), float(i1b),
                      float(i2x), float(i2y), float(i2b), float(theta_e)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="lens_microlensing",
    description="a microlensing event, exactly — top: the source star slides behind "
                "a stellar lens while its two unresolvable images (exact theta_pm, "
                "brightness ~ exact |mu_pm|, signed sum rule mu+ + mu- = 1 asserted) "
                "slide along the axis; bottom: the Paczynski light curve "
                "A(u) = (u^2+2)/(u sqrt(u^2+4)) revealed live — flat, smooth "
                "achromatic rise, peak at closest approach, symmetric fall. The "
                "signature OGLE/MOA hunt for; how worlds are found by the kink a "
                "planet adds to this curve. --frames runs one transit.",
    renderer=_render,
)
