"""The two horizons — what we can see versus what we can ever reach.

Λ splits the sky into three zones, and both boundaries are exact integrals of the
closed-form scale factor (``engine.cosmology``):

    D_PH(t) = c·∫₀ᵗ dt'/a(t')      the PARTICLE horizon — how far light has ever
                                    been able to come: ≈ 46 Gly comoving today
                                    (asserted) — the observable universe;
    D_EH(t) = c·∫ₜ^∞ dt'/a(t')     the EVENT horizon — how far light leaving NOW
                                    can ever reach: FINITE (≈ 16.7 Gly, asserted)
                                    purely because of Λ.

The scene plays 2 → 60 Gyr in comoving coordinates (galaxies pinned to the grid —
expansion is divided out, so the physics is all in the moving circles):

* the **cyan circle** (particle horizon) GROWS — ever more galaxies fade into
  view as their first light arrives;
* the **magenta circle** (event horizon) SHRINKS in comoving terms — galaxies it
  sweeps past flare amber and go red: from that moment, nothing we send will ever
  reach them, and their newest light will never reach us (we keep seeing their
  past, redshifting into a frozen goodbye);
* galaxies still inside both are bright and blue-white — mutual causal contact;
  outside the cyan circle they are dark — not yet seen;
* the ledgers track both horizons in Gly: cyan climbing toward its asymptote,
  magenta collapsing toward zero — in the far future only the local group remains.

The suite asserts both integrals against the real numbers, that the particle
horizon only grows, and that the comoving event horizon only shrinks. The sky is
slowly emptying — enjoy it while it's full. --frames runs 2 → 60 Gyr; iMouse pans.
See ``docs/research/56-expanding-universe.md`` (Part III).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.cosmology import event_horizon_gly, particle_horizon_gly
from ..scene import Scene

_T_CYCLE = 16.0
_T_START, _T_END = 2.0, 60.0
_N_GAL = 22
_SCALE = 1.20 / 50.0            # screen units per comoving Gly


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, gal: wp.array(dtype=wp.vec3), n_gal: int,
                   r_ph: float, r_eh: float,
                   ph_frac: float, eh_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    r_pix = wp.sqrt(x * x + y * y)
    wl = wp.max(0.006, 1.6 * px)

    # ---- comoving grid (expansion divided out): faint fixed rings ----
    wg = wp.max(0.004, 1.2 * px)
    for q in range(1, 5):
        rr = float(q) * 12.5 * 0.024
        d_r = wp.abs(r_pix - rr)
        col = col + wp.vec3(0.10, 0.13, 0.22) * (0.30 * wp.exp(-(d_r * d_r) / (wg * wg)))

    # ---- us ----
    wc = wp.max(0.016, 2.4 * px)
    col = col + wp.vec3(0.95, 0.90, 0.75) * (1.3 * wp.exp(-(r_pix * r_pix) / (wc * wc)))

    # ---- the galaxies: zone-tinted (gal = comoving x, y, flare) ----
    for g in range(n_gal):
        gx = gal[g][0] * 0.024
        gy = gal[g][1] * 0.024
        rg = wp.sqrt(gal[g][0] * gal[g][0] + gal[g][1] * gal[g][1])
        d2 = (x - gx) * (x - gx) + (y - gy) * (y - gy)
        wgl = wp.max(0.015, 2.3 * px)
        if rg <= r_ph:
            if rg <= r_eh:
                tintg = wp.vec3(0.60, 0.80, 1.00)       # mutual contact
                amp = 1.1
            else:
                tintg = wp.vec3(1.00, 0.42, 0.30)       # visible, forever unreachable
                amp = 0.85
            col = col + tintg * (amp * wp.exp(-d2 / (wgl * wgl)))
            # the goodbye flare as the event horizon sweeps past
            col = col + wp.vec3(1.00, 0.72, 0.25) * \
                (2.0 * gal[g][2] * wp.exp(-d2 / (wgl * wgl * 3.0)))

    # ---- the horizons ----
    d_ph = wp.abs(r_pix - r_ph * 0.024)
    col = col + wp.vec3(0.35, 0.85, 1.00) * (0.85 * wp.exp(-(d_ph * d_ph) / (wl * wl)))
    d_eh = wp.abs(r_pix - r_eh * 0.024)
    col = col + wp.vec3(1.00, 0.35, 0.80) * (0.85 * wp.exp(-(d_eh * d_eh) / (wl * wl)))

    # ---- the ledgers: PH (cyan, up) and EH (magenta, down from top) ----
    if x > 1.50 and x < 1.58 and y > -1.10 and y < -1.10 + 2.2 * ph_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.62 and x < 1.70 and y > -1.10 and y < -1.10 + 2.2 * eh_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0

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
            ang = 2.399963 * float(k) + 0.7
            rad = 6.0 + 42.0 * math.sqrt((float(k) + 0.5) / _N_GAL)   # comoving Gly
            pts.append((rad * math.cos(ang), rad * math.sin(ang) * 0.60))
        _GALS = np.asarray(pts, np.float32)
    return _GALS


def _render(width, height, time, mouse, device):
    tau_c = math.fmod(float(time), _T_CYCLE)
    s = tau_c / _T_CYCLE
    t_now = _T_START + s * (_T_END - _T_START)

    r_ph = particle_horizon_gly(t_now, n=1500)
    r_eh = event_horizon_gly(t_now, n=1500)

    gals = _galaxies()
    draw = np.zeros((_N_GAL, 3), np.float32)
    draw[:, 0] = gals[:, 0]
    draw[:, 1] = gals[:, 1]
    rg = np.hypot(gals[:, 0], gals[:, 1])
    # goodbye flare: bright just after the event horizon sweeps inside a galaxy
    behind = np.clip(rg - r_eh, 0.0, None)
    draw[:, 2] = np.exp(-behind / 1.6) * (behind > 0.0)

    # ledgers normalized to the animation's own range
    ph_frac = min(r_ph / 58.0, 1.0)
    eh_frac = min(r_eh / event_horizon_gly(_T_START, n=1500), 1.0)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array(draw, dtype=wp.vec3, device=device), int(_N_GAL),
                      float(r_ph), float(r_eh),
                      float(ph_frac), float(eh_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="cosmo_horizons",
    description="the two horizons in comoving coordinates, both exact integrals "
                "of the closed-form a(t): the cyan particle horizon "
                "c int dt'/a GROWS (46 Gly today, asserted — the observable "
                "universe) while the magenta event horizon c int_t^inf dt'/a "
                "SHRINKS (16.7 Gly today, asserted finite — pure Lambda): "
                "galaxies it sweeps past flare amber and go red — visible "
                "forever, reachable never — while blue-white ones remain in "
                "mutual contact; ledgers track both radii in Gly. The sky is "
                "slowly emptying. --frames runs 2 to 60 Gyr.",
    renderer=_render,
)
