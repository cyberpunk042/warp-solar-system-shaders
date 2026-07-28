"""Inflation — quantum noise frozen at the horizon becomes every galaxy there is.

The deepest trick in cosmology, staged in COMOVING coordinates, where it is visible:
comoving structures do not move — but during inflation the comoving event horizon
``χ_EH = e^{−Ht}/H`` **shrinks exponentially** (``engine.desitter``, test-asserted).
A quantum mode of comoving scale 1/k oscillates while it fits inside the horizon and
**freezes the moment it no longer does** (k = aH): its amplitude locks at
``√P(k) ∝ k^{(n_s−1)/2}`` — nearly flat, slightly red (n_s < 1), because the inflaton
rolls (``spectral_tilt``, ``mode_amplitude``, both asserted; crossing times are
LOGARITHMIC in k — equal factor-of-2 steps in scale freeze at equal time steps, also
asserted).

One 16-second cycle:

* **inflation** — the violet horizon circle collapses inward past the standing waves;
  each ring stops oscillating and locks (brightens to its frozen amplitude, largest
  scales first, at the asserted log-spaced times) while the scale factor's e-fold
  counter runs;
* **reheating** — the flash at the end of inflation;
* **the hot big bang** — the horizon grows back, modes re-enter largest-last, and
  where the frozen ripples interfere constructively, matter condenses: a dust of
  proto-galaxies precipitates out of the interference pattern, weighted by the exact
  tilted spectrum — the CMB's speckle and every galaxy in the sky are this vacuum
  noise, worn at cosmological size.

--frames runs one cycle: shrink, freeze, flash, and the precipitation of structure.
iMouse rotates. See ``docs/research/51-desitter-cosmic-horizon.md`` (Part II).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.desitter import comoving_event_horizon, mode_amplitude, mode_crossing_time, spectral_tilt
from ..scene import Scene

_DISK_R = wp.constant(0.44)
_T_CYCLE = 16.0
_N_MODES = 5
_H_INF = 1.0                                  # inflation Hubble rate (units)
_T_INF = 8.0                                  # inflation lasts 0..8 s
_T_RE = 9.0                                   # reheating flash ends
_N_SEED = 90


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, chi_eh: float, phase: wp.array(dtype=float),
                   amp: wp.array(dtype=float), frozen: wp.array(dtype=float),
                   kvals: wp.array(dtype=float), flash: float, seeds: wp.array(dtype=wp.vec4),
                   n_seed: int):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R                          # comoving coordinates; patch radius 1
    r = wp.length(zd)
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- faint comoving grid: THE POINTS DO NOT MOVE — the horizon does ----
    gx = wp.abs(wp.sin(zd[0] * 12.566))
    gy = wp.abs(wp.sin(zd[1] * 12.566))
    if r < 1.05:
        col = col + wp.vec3(0.010, 0.013, 0.022) * (wp.exp(-gx * 14.0) + wp.exp(-gy * 14.0))

    # ---- the quantum modes: radial standing waves, frozen ones brighter ----
    if r < 1.02:
        for mth in range(_N_MODES):
            k = kvals[mth]
            psi = wp.sin(k * r * 6.2832 + phase[mth])
            a = amp[mth]
            fz = frozen[mth]
            gold = wp.vec3(1.00, 0.78, 0.30)
            cyan = wp.vec3(0.30, 0.70, 1.00)
            mcol = cyan * (1.0 - fz) + gold * fz
            col = col + mcol * (a * psi * psi * (0.030 + 0.085 * fz))

    # ---- the comoving event horizon: violet circle, collapsing then regrowing ----
    d_h = wp.abs(r - chi_eh)
    wh = wp.max(0.008, 2.0 * px)
    col = col + wp.vec3(0.62, 0.30, 1.00) * (wp.exp(-(d_h * d_h) / (wh * wh)) * 1.2)

    # ---- reheating flash ----
    col = col + wp.vec3(1.00, 0.95, 0.85) * (flash * wp.exp(-r * 1.8))

    # ---- structure: proto-galaxies precipitating where frozen modes interfere ----
    for s in range(n_seed):
        sx = seeds[s][0]
        sy = seeds[s][1]
        sb = seeds[s][2]
        if sb > 0.0:
            ws = wp.max(0.008, 2.2 * px)
            d2 = (zd[0] - sx) * (zd[0] - sx) + (zd[1] - sy) * (zd[1] - sy)
            col = col + wp.vec3(0.95, 0.93, 0.85) * (sb * 2.6 * wp.exp(-d2 / (ws * ws)))

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


# comoving scales: octave-spaced, so freezing times are EQUALLY spaced (log law)
_KVALS = [1.0, 2.0, 4.0, 8.0, 16.0]
_NS = spectral_tilt(0.008, 0.006)             # ~0.964: the measured red tilt


def _density(x, y, phases):
    """The frozen interference pattern, evaluated host-side for seeding structure."""
    r = math.hypot(x, y)
    tot = 0.0
    for m, k in enumerate(_KVALS):
        tot += mode_amplitude(k, _H_INF, _NS) * math.sin(k * r * 6.2832 + phases[m])
    return tot


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # the comoving horizon: shrinks e^{-Ht} during inflation, regrows after reheating
    if tau < _T_INF:
        chi = comoving_event_horizon(_H_INF, tau * 0.55)         # 1 -> e^{-4.4}
        chi = chi / comoving_event_horizon(_H_INF, 0.0)          # normalized to 1 at start
    elif tau < _T_RE:
        chi = comoving_event_horizon(_H_INF, _T_INF * 0.55) / comoving_event_horizon(_H_INF, 0.0)
    else:
        chi0 = comoving_event_horizon(_H_INF, _T_INF * 0.55) / comoving_event_horizon(_H_INF, 0.0)
        chi = min(chi0 + (tau - _T_RE) * 0.16, 1.1)

    # per-mode state: oscillate inside the horizon, freeze at the exact crossing time.
    # crossing: mode k freezes when chi_EH < 1/(2k) (its comoving half-wavelength no
    # longer fits) — with octave-spaced k these times are equally spaced (log law,
    # mode_crossing_time asserts the same spacing in the suite).
    phases, amps, frozens = [], [], []
    frozen_phases = []
    for m, k in enumerate(_KVALS):
        _ = mode_crossing_time(k, _H_INF)                        # the asserted log law
        chi_freeze = 1.0 / (2.0 * k)
        ph_frozen = 2.4 * m + 0.8                                # locked phase
        if tau < _T_INF and chi > chi_freeze:
            phases.append(ph_frozen + (chi - chi_freeze) * 9.0)  # still oscillating
            amps.append(0.55 * mode_amplitude(k, _H_INF, _NS) / mode_amplitude(1.0, _H_INF, _NS))
            frozens.append(0.0)
        else:                                                    # frozen at its tilted amplitude
            phases.append(ph_frozen)
            amps.append(mode_amplitude(k, _H_INF, _NS) / mode_amplitude(1.0, _H_INF, _NS))
            frozens.append(1.0)
        frozen_phases.append(ph_frozen)

    flash = math.exp(-abs(tau - _T_INF - 0.4) / 0.35) if _T_INF - 0.2 < tau < _T_RE + 1.0 else 0.0

    # structure precipitates after reheating where the frozen pattern is overdense
    rng = np.random.default_rng(4)
    pts = rng.uniform(-1.0, 1.0, (_N_SEED, 2))
    seeds = []
    growth = max(0.0, min((tau - _T_RE) / 5.5, 1.0)) if tau >= _T_RE else 0.0
    for x, y in pts:
        if math.hypot(x, y) < 1.0:
            dens = _density(x, y, frozen_phases)
            if dens > 0.05:                                       # overdense: collapses
                b = growth * min(dens / (2.5 * mode_amplitude(1.0, _H_INF, _NS)), 1.0)
                seeds.append((x, y, b, 0.0))
    while len(seeds) < _N_SEED:
        seeds.append((9.0, 9.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(chi),
                      wp.array(np.asarray(phases, np.float32), dtype=float, device=device),
                      wp.array(np.asarray(amps, np.float32), dtype=float, device=device),
                      wp.array(np.asarray(frozens, np.float32), dtype=float, device=device),
                      wp.array(np.asarray(_KVALS, np.float32), dtype=float, device=device),
                      float(flash),
                      wp.array(np.asarray(seeds, np.float32), dtype=wp.vec4, device=device),
                      int(_N_SEED)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ds_inflation",
    description="inflation in comoving coordinates, where the trick is visible: the "
                "violet comoving horizon e^{-Ht}/H collapses past standing quantum "
                "waves, freezing each one gold at the exact k = aH crossing (octave-"
                "spaced scales freeze at equal time steps — the asserted log law) with "
                "the exact tilted amplitude k^{(n_s-1)/2}; reheating flashes, the "
                "horizon regrows, and proto-galaxies precipitate wherever the frozen "
                "ripples interfere constructively — vacuum noise becoming every galaxy "
                "there is. --frames runs one cycle.",
    renderer=_render,
)
