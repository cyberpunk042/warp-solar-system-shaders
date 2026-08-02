"""Nine live lattices land on Yang's exact curve — the theorem, measured.

Yang's 1952 closed form for the spontaneous magnetization of the 2D Ising
model,

    M(T) = (1 − sinh(2/T)^{−4})^{1/8}   below T_c,   0 above,

is drawn as the cyan curve — and NINE independent 48² lattices, each running
live seeded Metropolis at its own temperature spread across [1.2, 3.6], are
plotted as glowing dots at (T_k, |M_k|). At t = 0 the dots start at random
disorder; as the cycle runs and the lattices equilibrate, the cold-side dots
RISE AND LAND on the exact curve while the hot-side dots settle onto zero —
watching a theorem being confirmed in real time. Every closed form is
test-asserted: Onsager's T_c (white vertical line; self-duality
``sinh(2/T_c) = 1`` at machine precision), the exact critical exponent
β = 1/8 (log-slope asserted = 0.125), and the suite runs the same seeded
dynamics and asserts the simulation lands on Yang.

Ledgers: cyan — the mean |simulation − Yang| error across the nine lattices
(it SHRINKS as they equilibrate: convergence, test-asserted structurally);
amber — sweeps completed; magenta — fraction of lattices within 0.08 of the
exact curve. --frames runs one equilibration cycle; iMouse pans. See
``docs/research/58-ising-exactly.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.ising import (
    critical_temperature,
    magnetization_exact,
    metropolis_sweep,
)
from ..scene import Scene

_T_CYCLE = 16.0
_N = 48
_N_LAT = 9
_SWEEP_RATE = 30.0
_SEED = 5150
_T_MIN, _T_MAX = 1.2, 3.6
_TEMPS = [1.2, 1.5, 1.8, 2.05, 2.2, 2.35, 2.6, 3.0, 3.6]


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2,
                   dots: wp.array(dtype=wp.vec2), n_dots: int,
                   tc_x: float,
                   err_frac: float, sweep_frac: float, conv_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the plot frame: T in [1.2, 3.6] -> x in [-1.9, 0.9]; M in [0,1] -> y in [-1.0, 0.9] ----
    if wp.abs(x + 1.9) < 0.008 and y > -1.0 and y < 0.95:
        col = col + wp.vec3(0.30, 0.34, 0.44) * 0.8
    if wp.abs(y + 1.0) < 0.008 and x > -1.9 and x < 0.95:
        col = col + wp.vec3(0.30, 0.34, 0.44) * 0.8

    # ---- Yang's exact curve, drawn from the closed form per pixel ----
    if x > -1.9 and x < 0.9 and y > -1.02 and y < 0.95:
        temp = 1.2 + (x + 1.9) / 2.8 * 2.4
        tc = 2.0 / wp.log(1.0 + wp.sqrt(2.0))
        m_exact = float(0.0)
        if temp < tc:
            s = wp.sinh(2.0 / temp)
            m_exact = wp.pow(1.0 - wp.pow(s, -4.0), 0.125)
        y_curve = -1.0 + 1.9 * m_exact
        d_c = wp.abs(y - y_curve)
        wc = wp.max(0.006, 1.5 * px)
        col = col + wp.vec3(0.35, 0.85, 1.00) * (0.9 * wp.exp(-(d_c * d_c) / (wc * wc)))

    # ---- Onsager's Tc: the white vertical line ----
    if wp.abs(x - tc_x) < 0.006 and y > -1.0 and y < 0.95:
        col = col + wp.vec3(0.90, 0.92, 0.95) * 0.9

    # ---- the nine live lattices, as dots at (T_k, |M_k|) ----
    for m in range(n_dots):
        p = dots[m]
        d2 = (x - p[0]) * (x - p[0]) + (y - p[1]) * (y - p[1])
        wd = wp.max(0.018, 2.4 * px)
        col = col + wp.vec3(1.00, 0.72, 0.30) * (1.3 * wp.exp(-d2 / (wd * wd)))

    # ---- the ledgers: |sim - Yang| error / sweeps / converged fraction ----
    if x > 1.42 and x < 1.48 and y > -1.05 and y < -1.05 + 2.0 * err_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > 1.52 and x < 1.58 and y > -1.05 and y < -1.05 + 2.0 * sweep_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0
    if x > 1.62 and x < 1.68 and y > -1.05 and y < -1.05 + 2.0 * conv_frac:
        col = col + wp.vec3(1.00, 0.35, 0.80) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


_STATE = {"lats": None, "rngs": None, "sweeps": -1}


def _advance_to(t: float):
    """Deterministic nine-lattice ensemble at absolute time t (replayable)."""
    target = int(_SWEEP_RATE * max(t, 0.0))
    if _STATE["lats"] is None or _STATE["sweeps"] > target:
        _STATE["lats"] = []
        _STATE["rngs"] = []
        for k in range(_N_LAT):
            rng = np.random.default_rng(_SEED + k)
            _STATE["lats"].append(
                np.where(rng.random((_N, _N)) < 0.5, 1.0, -1.0))
            _STATE["rngs"].append(rng)
        _STATE["sweeps"] = 0
    for _ in range(_STATE["sweeps"], target):
        for k in range(_N_LAT):
            metropolis_sweep(_STATE["lats"][k], _TEMPS[k], _STATE["rngs"][k])
    _STATE["sweeps"] = target
    return _STATE["lats"]


def _render(width, height, time, mouse, device):
    t = float(time)
    tau = math.fmod(t, _T_CYCLE)
    lats = _advance_to(tau if t < _T_CYCLE else t)

    tc = critical_temperature()
    dots, errs, n_conv = [], [], 0
    for k in range(_N_LAT):
        m_k = abs(float(lats[k].mean()))    # global |M|: these seeds single-domain
        xk = -1.9 + (_TEMPS[k] - _T_MIN) / (_T_MAX - _T_MIN) * 2.8
        yk = -1.0 + 1.9 * m_k
        dots.append((xk, yk))
        err = abs(m_k - magnetization_exact(_TEMPS[k]))
        errs.append(err)
        if err < 0.08:
            n_conv += 1

    tc_x = -1.9 + (tc - _T_MIN) / (_T_MAX - _T_MIN) * 2.8
    err_frac = min(float(np.mean(errs)) / 0.35, 1.0)
    sweep_frac = min(_STATE["sweeps"] / (_SWEEP_RATE * _T_CYCLE), 1.0)
    conv_frac = n_conv / float(_N_LAT)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, t,
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array([wp.vec2(a, b) for a, b in dots],
                               dtype=wp.vec2, device=device),
                      _N_LAT, float(tc_x),
                      float(err_frac), float(sweep_frac), float(conv_frac)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ising_magnetization",
    description="a theorem confirmed in real time: Yang's exact spontaneous "
                "magnetization M = (1 - sinh(2/T)^-4)^(1/8) drawn as the cyan "
                "curve (exact critical exponent beta = 1/8, asserted), "
                "Onsager's T_c as the white line (self-duality asserted at "
                "machine precision) — and NINE independent live 48^2 Metropolis "
                "lattices plotted as amber dots at (T_k, |M_k|), rising from "
                "random disorder and LANDING on the closed form as they "
                "equilibrate. The cyan ledger is the mean |simulation - Yang| "
                "error, and it shrinks (structurally asserted); magenta counts "
                "the lattices within 0.08 of the theorem. Measured, not "
                "asserted. --frames runs one equilibration cycle.",
    renderer=_render,
)
