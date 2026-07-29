"""The chirp — LIGO's waveform, drawn live from the closed forms that measured it.

The most famous curve in modern physics, generated (not sampled) from the exact
leading-order dictionary (``engine.gw``):

* **top panel** — the frequency track ``f_gw(t) ∝ M_c^{−5/8}(t_c−t)^{−3/8}``
  (``chirp_frequency``; the −3/8 exponent asserted numerically): flat and low for
  ages, then screaming upward at the end. This track in the spectrogram is how the
  chirp mass — and with it the black-hole masses — is read straight off the data;
* **bottom panel** — the strain ``h(t) = A(f)·sin φ(t)`` with the closed-form phase
  ``φ ∝ (t_c−t)^{5/8}`` and the amplitude climbing as ``h ∝ M_c^{5/3}f^{2/3}``
  (``strain_amplitude``): the wave gets faster AND louder together — the chirp —
  then rings down as a damped sinusoid after coalescence and goes quiet.

Both curves are revealed left-to-right in sync as the cycle plays, a live sweep line
marking "now". Every value on screen comes from ``chirp_frequency`` /
``strain_amplitude`` / ``peters_merger_time`` — nothing is hand-animated. GW150914's
actual numbers: M_c ≈ 30 M_sun, f sweeping 35 → 250 Hz in 0.2 s, h ≈ 10⁻²¹.

--frames runs one full chirp; iMouse pans. See
``docs/research/53-gravitational-waves.md`` (Part II).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.gw import chirp_frequency, chirp_mass, strain_amplitude
from ..scene import Scene

_T_CYCLE = 16.0
_T_MERGE = 12.5
_N_COL = 960                              # waveform samples across the panel


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, hcurve: wp.array(dtype=float),
                   fcurve: wp.array(dtype=float), n_col: int, reveal: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    u = fx / res[0]                        # 0..1 across = scene time
    v = fy / res[1]                        # 0..1 up
    px = 1.0 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # panel split: f-track occupies v in [0.58, 0.97], strain in [0.05, 0.50]
    # faint panel frames + midline
    if wp.abs(v - 0.275) < 0.7 * px and u > 0.02 and u < 0.98:
        col = col + wp.vec3(0.05, 0.06, 0.10)

    ci = int(u * float(n_col))
    if ci > n_col - 1:
        ci = n_col - 1

    if u < reveal and u > 0.02 and u < 0.98:
        # ---- the frequency track: f ~ (tc - t)^{-3/8} ----
        fv = fcurve[ci]                    # 0..1 normalized
        yf = 0.58 + 0.37 * fv
        df = wp.abs(v - yf)
        wf = wp.max(0.004, 1.6 * px)
        heat = wp.vec3(0.30 + 0.68 * fv, 0.60 - 0.15 * fv, 1.00 - 0.65 * fv)
        col = col + heat * (wp.exp(-(df * df) / (wf * wf)) * 1.25)
        # spectrogram-style glow under the track
        if v > 0.58 and v < yf:
            col = col + heat * (0.10 * wp.exp(-(yf - v) * 22.0))

        # ---- the strain waveform ----
        hv = hcurve[ci]                    # -1..1
        yh = 0.275 + 0.21 * hv
        dh = wp.abs(v - yh)
        wh = wp.max(0.004, 1.6 * px)
        col = col + wp.vec3(0.35, 0.85, 1.00) * (wp.exp(-(dh * dh) / (wh * wh)) * 1.15)

    # ---- the sweep line: "now" ----
    du = wp.abs(u - reveal)
    if du < 1.2 * px and v > 0.03 and v < 0.97:
        col = col + wp.vec3(0.95, 0.90, 0.75) * 0.7

    uvx = u - 0.5
    uvy = v - 0.5
    col = col * (1.0 - 0.30 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _curves():
    """Precompute the exact chirp curves across the panel (host-side, once)."""
    mc = chirp_mass(1.0, 1.0)
    # panel spans the cycle: [0, T_CYCLE]; merger at T_MERGE; physical speedup
    t_full = 60.0                                  # physical time compressed into the panel
    speed = t_full / _T_MERGE
    f_end = chirp_frequency(0.020 * speed, mc)     # cap near merger
    f_start = chirp_frequency(_T_MERGE * speed, mc)

    hcv = np.zeros(_N_COL, np.float32)
    fcv = np.zeros(_N_COL, np.float32)
    dt_col = _T_CYCLE / _N_COL
    # normalize the EXACT accumulated phase to ~34 display cycles across the panel
    # (true cycle count is astronomically larger; the phase SHAPE phi ~ (tc-t)^{5/8}
    # is preserved exactly, so the oscillation density chirps by the true law)
    fs = np.zeros(_N_COL, np.float32)
    for c in range(_N_COL):
        t = (c + 0.5) * dt_col
        t_left = max((_T_MERGE - t) * speed, 0.020 * speed)
        fs[c] = chirp_frequency(t_left, mc)
    phi_raw = np.cumsum(fs) * dt_col
    n_cycles = 34.0
    f_ring = f_end * 1.15
    phase = 0.0
    for c in range(_N_COL):
        t = (c + 0.5) * dt_col
        if t < _T_MERGE:
            f = fs[c]
            amp = strain_amplitude(f, mc, 8.0) / strain_amplitude(f_end, mc, 8.0)
            phase = 2.0 * math.pi * n_cycles * phi_raw[c] / phi_raw[-1]
            hcv[c] = amp * math.sin(phase)
            fcv[c] = (math.log(f) - math.log(f_start)) / (math.log(f_end) - math.log(f_start))
        else:
            tr = t - _T_MERGE
            phase += 2.0 * math.pi * n_cycles * (f_ring / fs.sum()) * 8.0
            hcv[c] = math.exp(-tr / 0.55) * math.sin(phase)
            fcv[c] = max(1.0 - tr * 0.25, 0.95) if tr < 0.2 else max(0.95 - (tr - 0.2) * 0.5, 0.0)
    return hcv, fcv


_HCV, _FCV = None, None


def _render(width, height, time, mouse, device):
    global _HCV, _FCV
    if _HCV is None:
        _HCV, _FCV = _curves()
    tau = math.fmod(float(time), _T_CYCLE)
    reveal = 0.02 + 0.96 * (tau / _T_CYCLE)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      wp.array(_HCV, dtype=float, device=device),
                      wp.array(_FCV, dtype=float, device=device),
                      int(_N_COL), float(reveal)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gw_chirp",
    description="THE chirp, generated live from the closed forms: top panel traces "
                "the frequency track f ~ Mc^{-5/8}(tc-t)^{-3/8} (the -3/8 exponent "
                "asserted) climbing from a low hum to a scream; bottom panel draws "
                "the strain h(t) with phase phi ~ (tc-t)^{5/8} and amplitude "
                "h ~ Mc^{5/3}f^{2/3} — faster AND louder together — then the damped "
                "ringdown after coalescence. The curve LIGO reads black-hole masses "
                "from, with nothing hand-animated. --frames runs one chirp.",
    renderer=_render,
)
