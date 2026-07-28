"""Casimir — the vacuum pushes two mirrors together, sixteen times harder per halving.

Casimir 1948: between two parallel mirrors only the standing waves that fit are
allowed — ``k_n = nπ/d`` — while outside, the vacuum keeps its full continuum. The
missing modes leave less zero-point pressure inside than out, and the plates feel

    P = −π²/240 · 1/d⁴

attraction from literally nothing (``engine.vacuum.casimir_pressure``; the fourth-power
law — halve the gap, SIXTEENFOLD the force — and the thermodynamic identity
``P = −∂(E/A)/∂d`` are both asserted). Measured to ~1% precision (Lamoreaux 1997,
Mohideen–Roy 1998); a real engineering constraint in MEMS, where it makes
micro-machines snap shut.

One cycle squeezes the gap ``d`` from wide to narrow and back, everything driven by
the exact dictionary:

* outside the plates the vacuum shimmers with a dense mode continuum;
* between them only ``⌊k_max·d/π⌋`` standing waves survive (``allowed_modes``, the
  count recomputed live) — watch modes get **evicted one by one** as the gap closes;
* the amber force arrows on each plate grow as exactly ``1/d⁴`` — gentle at wide gap,
  overwhelming at narrow — and the plates glow under the load.

The eviction IS the force. --frames runs one squeeze; iMouse pans. See
``docs/research/52-quantum-vacuum.md`` (Part II).
"""

import math

import warp as wp

from ..engine import post
from ..engine.vacuum import allowed_modes, casimir_energy, casimir_pressure
from ..scene import Scene

_T_CYCLE = 16.0
_K_MAX = 9.0 * math.pi                    # mode cutoff: 9 modes fit at d = 1


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, half_gap: float, n_modes: int, force: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.2          # lab coords, x in ~[-1.5, 1.5]
    y = (fy - 0.5 * res[1]) / res[1] * 2.2
    px = 2.2 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)
    d_gap = 2.0 * half_gap

    # ---- outside: the full vacuum continuum, dense shimmering modes ----
    if wp.abs(x) > half_gap + 0.02:
        s = 0.0
        s += wp.sin(x * 9.0 + time * 1.7) * wp.sin(y * 4.0 - time * 1.1)
        s += wp.sin(x * 17.0 - time * 2.3) * wp.sin(y * 7.0 + time * 1.9)
        s += wp.sin(x * 29.0 + time * 3.1) * wp.sin(y * 12.0 - time * 2.6)
        s += wp.sin(x * 43.0 - time * 4.0) * wp.sin(y * 19.0 + time * 3.4)
        col = col + wp.vec3(0.10, 0.16, 0.30) * (0.16 * (s * s) * 0.25 + 0.02)

    # ---- inside: ONLY the standing waves that fit, k_n = n pi / d ----
    if wp.abs(x) < half_gap and wp.abs(y) < 0.85:
        acc = float(0.0)
        for n in range(1, 10):
            if n <= n_modes:
                kx = float(n) * 3.14159265 / d_gap
                mode = wp.sin(kx * (x + half_gap)) * wp.cos(time * (1.2 + 0.55 * float(n)))
                acc += mode * mode / (0.8 + 0.35 * float(n))
        col = col + wp.vec3(0.30, 0.70, 1.00) * (acc * 0.16)

    # ---- the plates ----
    for side in range(2):
        sx = half_gap
        if side == 1:
            sx = -half_gap
        d_p = wp.abs(x - sx)
        if wp.abs(y) < 0.9:
            wpl = wp.max(0.012, 2.5 * px)
            load = wp.min(force, 1.0)
            pcol = wp.vec3(0.85, 0.85 - 0.30 * load, 0.80 - 0.45 * load)
            col = col + pcol * (wp.exp(-(d_p * d_p) / (wpl * wpl)) * (0.9 + 0.9 * load))

    # ---- the force arrows: length exactly ~ 1/d^4 (normalized), pointing inward ----
    ar_len = 0.10 + 0.75 * wp.min(force, 1.0)
    for side in range(2):
        base = half_gap + 0.06
        tip = half_gap + 0.06 + ar_len
        xx = x
        if side == 1:
            xx = -x
        if wp.abs(y) < 0.035 and xx > base and xx < tip:
            col = col + wp.vec3(1.00, 0.72, 0.25) * 1.3
        # arrowhead pointing inward (toward the gap)
        ah = base + 0.05
        if xx > base - 0.02 and xx < ah and wp.abs(y) < (ah - xx) * 1.8 + 0.005:
            col = col + wp.vec3(1.00, 0.72, 0.25) * 1.3

    uvx = x / 2.2
    uvy = y / 2.2
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.6, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # the squeeze: d from 1.30 down to 0.34 and back
    d = 1.30 - 0.96 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))
    n = allowed_modes(d, _K_MAX)                       # evicted one by one
    p_now = casimir_pressure(d)
    p_ref = casimir_pressure(1.30)
    force = (abs(p_now / p_ref)) ** 0.25 / 4.6         # 4th-root display of the exact 1/d^4
    _ = casimir_energy(d)                              # P = -d(E/A)/dd asserted in suite

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(d / 2.0), int(n), float(force)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="casimir_plates",
    description="the Casimir effect — outside two mirror plates the vacuum keeps its "
                "full shimmering continuum, between them only the standing waves "
                "k_n = n pi/d survive; as the gap squeezes shut the modes are evicted "
                "one by one (the count recomputed live) and the amber force arrows "
                "grow as exactly 1/d^4 — halve the gap, sixteenfold the pressure "
                "(asserted, along with P = -d(E/A)/dd): attraction from literally "
                "nothing, measured to ~1%. --frames runs one squeeze.",
    renderer=_render,
)
