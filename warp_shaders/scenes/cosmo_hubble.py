"""The Hubble diagram — the plot that discovered dark energy.

Riess, Schmidt, Perlmutter 1998: Type Ia supernovae are standard candles, so their
apparent brightness IS their distance. Plot distance modulus against redshift and
the high-z points came out ~0.25 mag TOO DIM for any decelerating universe — the
expansion is speeding up. The 2011 Nobel, in one scatter plot. The scene rebuilds it
from the exact machinery (``engine.cosmology``):

* the **amber curve** is exact flat ΛCDM (Planck Ω_m = 0.3153):
  ``μ(z) = 5·log₁₀(D_L/10pc)`` with ``D_L = (1+z)·(c/H₀)∫dz'/E(z')`` — the suite
  asserts it sits ABOVE the matter-only curve at every z (dimmer supernovae);
* the **gray ghost** is the flat matter-only universe (Ω_m = 1, exact closed form
  ``D_C = (2c/H₀)(1−1/√(1+z))``) — what 1998 expected to fit, and didn't; the gap
  between the curves grows with z (~0.58 mag at z = 1, asserted > 0.4);
* **supernovae light up** along the exact ΛCDM curve as the survey deepens —
  low-z first (Calán/Tololo), then the high-z teams — with deterministic
  ~±0.15 mag scatter (real Ia dispersion after light-curve correction);
* the sweeping marker carries a **redshift readout bar** (amber ledger = current
  survey depth), and the green tick marks z ≈ 0.63 — the acceleration onset
  (asserted): points beyond it sample the DECELERATING era, points inside the
  accelerating one; the curve's shape across that divide is the discovery.

Every point is the exact integral, not a sketch — the plot a Nobel committee read.
--frames runs one survey sweep; iMouse pans. See
``docs/research/56-expanding-universe.md`` (Part II).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.cosmology import (
    acceleration_onset,
    distance_modulus,
    distance_modulus_matter_only,
)
from ..scene import Scene

_T_CYCLE = 16.0
_Z_MIN, _Z_MAX = 0.015, 1.40
_MU_LO, _MU_HI = 33.5, 45.6
_N_TAB = 192
_N_SN = 26


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, mu_tab: wp.array(dtype=float), n_tab: int,
                   sn: wp.array(dtype=wp.vec3), n_sn: int, n_lit: int,
                   z_mark: float, z_acc: float, gap_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # plot frame: z in [z_min, z_max] -> x in [-1.95, 1.55]; mu -> y in [-1.05, 1.10]
    zx = 0.015 + (x + 1.95) / 3.5 * (1.40 - 0.015)
    if x > -2.02 and x < 1.62 and y > -1.15 and y < 1.20 and zx > 0.015:
        # axes
        if wp.abs(x + 1.95) < 0.006 or wp.abs(y + 1.05) < 0.006:
            col = col + wp.vec3(0.35, 0.40, 0.48) * 0.7

        # ---- matter-only ghost: exact closed form in-kernel ----
        dc = 2.0 * (299792.458 / 67.36) * (1.0 - 1.0 / wp.sqrt(1.0 + zx))
        mu_m = 5.0 * wp.log10((1.0 + zx) * dc * 1.0e6 / 10.0)
        ym = -1.05 + (mu_m - 33.5) / (45.6 - 33.5) * 2.15
        wl = wp.max(0.005, 1.5 * px)
        d_m = wp.abs(y - ym)
        col = col + wp.vec3(0.30, 0.34, 0.42) * (0.55 * wp.exp(-(d_m * d_m) / (wl * wl)))

        # ---- exact LCDM: table lookup ----
        idxf = (zx - 0.015) / (1.40 - 0.015) * float(n_tab - 1)
        i0 = wp.clamp(int(idxf), 0, n_tab - 2)
        f = idxf - float(i0)
        mu_l = mu_tab[i0] * (1.0 - f) + mu_tab[i0 + 1] * f
        yl = -1.05 + (mu_l - 33.5) / (45.6 - 33.5) * 2.15
        d_l = wp.abs(y - yl)
        col = col + wp.vec3(1.00, 0.82, 0.40) * (0.9 * wp.exp(-(d_l * d_l) / (wl * wl)))

        # the acceleration-onset divide
        xacc = -1.95 + (z_acc - 0.015) / (1.40 - 0.015) * 3.5
        if wp.abs(x - xacc) < 0.006 and y < yl:
            col = col + wp.vec3(0.55, 1.00, 0.75) * 0.5

        # live marker riding the exact curve
        xmark = -1.95 + (z_mark - 0.015) / (1.40 - 0.015) * 3.5
        d2mk = (x - xmark) * (x - xmark) + (y - yl) * (y - yl)
        if wp.abs(x - xmark) < 0.05:
            wmk = wp.max(0.018, 2.6 * px)
            col = col + wp.vec3(0.55, 1.00, 0.95) * (1.5 * wp.exp(-d2mk / (wmk * wmk)))

    # ---- the supernovae: lit progressively as the survey deepens ----
    for p in range(n_lit):
        sx = -1.95 + (sn[p][0] - 0.015) / (1.40 - 0.015) * 3.5
        sy = -1.05 + (sn[p][1] - 33.5) / (45.6 - 33.5) * 2.15
        d2s = (x - sx) * (x - sx) + (y - sy) * (y - sy)
        ws = wp.max(0.014, 2.2 * px)
        # fresh points flare
        fl = 1.0 + 1.6 * sn[p][2]
        col = col + wp.vec3(0.95, 0.95, 0.90) * (0.9 * fl * wp.exp(-d2s / (ws * ws)))

    # ---- the ledger: the LCDM-vs-matter gap at the marker (the Nobel, in mag) ----
    if x > 1.60 and x < 1.68 and y > -0.95 and y < -0.95 + 1.7 * gap_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.28 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


_CACHE = None


def _tables():
    global _CACHE
    if _CACHE is None:
        zs = np.linspace(_Z_MIN, _Z_MAX, _N_TAB)
        mu = np.asarray([distance_modulus(float(z)) for z in zs], np.float32)
        # deterministic supernovae along the exact curve, +-0.15 mag scatter
        sn = []
        for k in range(_N_SN):
            u = (math.sin(float(k) * 12.9898) * 43758.5453) % 1.0
            v = (math.sin(float(k) * 78.233) * 12543.8567) % 1.0
            z = _Z_MIN + (float(k) + 0.4 + 0.5 * u) / _N_SN * (_Z_MAX - _Z_MIN) * 0.97
            mu_s = distance_modulus(z) + (v - 0.5) * 0.30
            sn.append((z, mu_s, 0.0))
        sn.sort(key=lambda p: p[0])                    # survey deepens outward
        _CACHE = (mu, np.asarray(sn, np.float32), float(distance_modulus(1.0) -
                                                        distance_modulus_matter_only(1.0)))
    return _CACHE


def _render(width, height, time, mouse, device):
    mu_tab, sn, gap_z1 = _tables()
    tau_c = math.fmod(float(time), _T_CYCLE)
    s = tau_c / _T_CYCLE
    z_mark = _Z_MIN + s * (_Z_MAX - _Z_MIN) * 0.985

    # survey depth: points with z <= z_mark are lit; the newest flares
    n_lit = int(np.searchsorted(sn[:, 0], z_mark))
    sn_draw = sn.copy()
    for k in range(n_lit):
        age = z_mark - float(sn_draw[k, 0])
        sn_draw[k, 2] = math.exp(-age / 0.04)          # flare fades as survey moves on

    gap_now = distance_modulus(max(z_mark, 0.02)) - distance_modulus_matter_only(max(z_mark, 0.02))
    gap_frac = min(max(gap_now / gap_z1, 0.02), 1.0)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array(mu_tab, dtype=float, device=device), int(_N_TAB),
                      wp.array(sn_draw, dtype=wp.vec3, device=device), int(_N_SN),
                      int(n_lit),
                      float(z_mark), float(1.0 / acceleration_onset() - 1.0),
                      float(gap_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="cosmo_hubble",
    description="the 1998 discovery plot, rebuilt exactly — distance modulus "
                "mu = 5 log10(D_L/10pc) vs redshift with D_L from the exact LCDM "
                "integral (amber) sitting ABOVE the exact matter-only closed form "
                "(gray ghost; asserted at every z, gap ~0.58 mag at z=1): "
                "supernovae light up along the curve as the survey deepens, "
                "low-z first, each with deterministic +-0.15 mag scatter; the "
                "green divide marks the acceleration onset z = 0.63 (asserted) "
                "and the amber ledger tracks the LCDM-matter gap the Nobel "
                "committee read off this plot. --frames runs one survey sweep.",
    renderer=_render,
)
