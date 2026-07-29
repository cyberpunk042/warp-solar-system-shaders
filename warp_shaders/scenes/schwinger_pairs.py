"""Schwinger — crank the field past critical and the vacuum short-circuits.

Sauter 1931, Schwinger 1951: a static electric field tugs on the virtual e⁺e⁻ pairs
that flicker in the vacuum. When the work done separating a pair across its Compton
wavelength rivals its rest mass — at the critical field ``E_c = m²/e ≈ 1.3×10¹⁸ V/m``
— the pairs stop being virtual. The rate per volume,

    Γ ∝ E² · exp(−π E_c / E),

is non-perturbative and brutally thresholded: at E = E_c/10 the exponential is
``e^{−10π} ≈ 10⁻¹⁴`` of critical (asserted, ``engine.vacuum.schwinger_rate``) —
essentially exact zero — and above E_c it is an avalanche. The vacuum behaves like a
dielectric with a breakdown voltage: exceed it and empty space conducts, the created
pairs screening the very field that made them until it collapses. (Next-generation
lasers — ELI, XCELS — are built to chase exactly this edge.)

One cycle charges the capacitor and lets the vacuum answer:

* the field between the electrodes ramps toward and past ``E_c`` — field lines
  brighten and tighten with E;
* pair events fire at the **exact rate curve**: nothing, nothing, nothing... then a
  drizzle near ~0.6 E_c, then the avalanche — each event a cyan electron and magenta
  positron born together and torn apart along the field toward opposite electrodes;
* past critical, the avalanche **shorts the gap**: a breakdown flash, the field
  collapses, and the cycle re-arms in a quiet vacuum.

--frames runs one charge-breakdown cycle; iMouse pans. See
``docs/research/52-quantum-vacuum.md`` (Part III).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.vacuum import schwinger_critical_field, schwinger_rate
from ..scene import Scene

_T_CYCLE = 16.0
_T_BOOM = 12.4                            # breakdown moment
_N_PAIR = 40


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, e_frac: float, boom: float,
                   pairs: wp.array(dtype=wp.vec4), n_pairs: int):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.2
    y = (fy - 0.5 * res[1]) / res[1] * 2.2
    px = 2.2 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the electrodes: anode above (+), cathode below (-) ----
    for side in range(2):
        ey = 0.78
        if side == 1:
            ey = -0.78
        d_e = wp.abs(y - ey)
        if wp.abs(x) < 1.05:
            we = wp.max(0.014, 2.5 * px)
            ecol = wp.vec3(1.00, 0.55, 0.25)
            if side == 1:
                ecol = wp.vec3(0.35, 0.65, 1.00)
            col = col + ecol * (wp.exp(-(d_e * d_e) / (we * we)) * (0.7 + 0.8 * e_frac))

    # ---- the field lines: vertical, brightness and density ~ E ----
    if wp.abs(y) < 0.74 and wp.abs(x) < 1.0:
        lines = wp.abs(wp.sin(x * (6.0 + 14.0 * e_frac) * 3.14159265))
        col = col + wp.vec3(0.55, 0.45, 0.90) * (wp.exp(-lines * 9.0) * (0.06 + 0.85 * e_frac * e_frac))

    # ---- the pairs: cyan electron up... no — electron to anode (up), positron down ----
    for p in range(n_pairs):
        bx = pairs[p][0]
        by = pairs[p][1]
        sep = pairs[p][2]                    # how far the two have been torn apart
        br = pairs[p][3]
        if br > 0.0:
            wq = wp.max(0.010, 2.6 * px)
            d2e = (x - bx) * (x - bx) + (y - (by + sep)) * (y - (by + sep))
            col = col + wp.vec3(0.30, 0.85, 1.00) * (br * 1.7 * wp.exp(-d2e / (wq * wq)))
            d2p = (x - bx) * (x - bx) + (y - (by - sep)) * (y - (by - sep))
            col = col + wp.vec3(1.00, 0.35, 0.75) * (br * 1.7 * wp.exp(-d2p / (wq * wq)))
            # the tearing filament between them
            if wp.abs(x - bx) < 0.006 and wp.abs(y - by) < sep:
                col = col + wp.vec3(0.80, 0.70, 1.00) * (br * 0.5)

    # ---- breakdown flash: the vacuum shorts out ----
    col = col + wp.vec3(1.00, 0.96, 0.88) * (boom * wp.exp(-(x * x + y * y) * 1.4))

    uvx = x / 2.2
    uvy = y / 2.2
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.6, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)
    ec = schwinger_critical_field()

    # the charge-up: E ramps toward 1.35 E_c, then breakdown dumps it
    if tau < _T_BOOM:
        e_field = ec * 1.35 * (tau / _T_BOOM) ** 1.6
        boom = 0.0
    else:
        e_field = ec * 1.35 * math.exp(-(tau - _T_BOOM) / 0.5)
        boom = math.exp(-(tau - _T_BOOM) / 0.45)
    e_frac = min(e_field / (1.35 * ec), 1.0)

    # pair events from the EXACT rate curve: integrate Gamma(E(t)) into a schedule
    rate_now = schwinger_rate(e_field, ec) / schwinger_rate(1.35 * ec, ec)
    rng = np.random.default_rng(23)
    births = []
    tt, acc = 0.0, 0.0
    while tt < _T_BOOM + 0.6 and len(births) < 400:
        ef = ec * 1.35 * (min(tt, _T_BOOM) / _T_BOOM) ** 1.6
        r = schwinger_rate(ef, ec) / schwinger_rate(1.35 * ec, ec)
        acc += r * 3.2 * 0.05                        # dt = 0.05, peak ~3.2 events/s... scaled
        if acc >= 1.0:
            acc -= 1.0
            births.append(tt)
        tt += 0.05
    _ = rate_now

    # fixed per-birth positions (drawn once, in birth order, so frames agree)
    locs = [(float(rng.uniform(-0.9, 0.9)), float(rng.uniform(-0.45, 0.45))) for _ in births]
    pairs = []
    for b, (bx, by) in zip(births, locs):
        if tau >= b and len(pairs) < _N_PAIR:
            age = tau - b
            if age < 1.6:
                sep = min(0.5 * age * age * (0.4 + e_frac), 0.7)     # torn apart, accelerating
                br = max(0.0, 1.0 - age / 1.6)
                pairs.append((bx, by, sep, br))
    while len(pairs) < _N_PAIR:
        pairs.append((9.0, 9.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(e_frac), float(boom),
                      wp.array(np.asarray(pairs, np.float32), dtype=wp.vec4, device=device),
                      int(_N_PAIR)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=6)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="schwinger_pairs",
    description="Schwinger pair production — the field between two electrodes ramps "
                "toward the critical E_c = m^2/e while the vacuum answers at the "
                "exact rate E^2 exp(-pi E_c/E): nothing, nothing, a drizzle near "
                "0.6 E_c, then the avalanche — cyan electrons and magenta positrons "
                "torn out of empty space along the field lines — until the created "
                "charge shorts the gap: breakdown flash, field collapse, quiet. "
                "The vacuum has a breakdown voltage. --frames runs one cycle.",
    renderer=_render,
)
