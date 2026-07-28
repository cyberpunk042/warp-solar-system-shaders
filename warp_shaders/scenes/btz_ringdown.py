"""Quasinormal ringdown — the black hole that rings in exact closed form.

Kick a black hole and it rings: damped oscillations at its **quasinormal frequencies**,
the fingerprint LIGO hears when merged holes settle. For almost every black hole those
frequencies are known only numerically — but BTZ is the exception. Its QNM spectrum is
EXACT (Birmingham-Sachs-Solodukhin 2001, host-side ``btz_qnm``, test-asserted):

    ω = ±k − 4πi·T·(n + Δ/2)

for boundary operator dimension Δ, angular momentum k, overtone n. Three theorems in one
line: the ringing frequency is just the momentum; the damping is set ONLY by the
temperature; overtones are spaced exactly ``4πT`` apart. And through the dictionary these
are precisely the **poles of the boundary CFT's retarded thermal correlator**: the rate
at which the horizon settles IS the rate at which the dual plasma thermalizes
(Horowitz-Hubeny). Ringdown = thermalization — one number, two languages.

The scene kicks the hole once per cycle and lets the exact spectrum do the rest:

* the horizon circle deforms in a k = 2 quadrupole, ringing at ω_re = k and damping at
  ``e^{ω_im·t}`` with ``ω_im = −4πT(0 + Δ/2)`` — the fundamental mode, drawn as the
  breathing outline;
* ripples of the perturbation **propagate outward** and drain into the boundary,
  carrying the same damped phase — the bulk field ``e^{ω_im·t_ret}·cos(kφ − ω_re·t_ret)``
  painted along the retarded time;
* the boundary ring flickers with the arriving signal and settles as ``e^{ω_im·t}`` —
  the dual CFT thermalizing at exactly the QNM rate — until the disk is quiet and the
  next kick lands.

The dispersion and damping are the honest closed forms; the wave rendering is a
retarded-time visualization at fixed propagation speed. See
``docs/research/48-btz-black-hole-on-the-disk.md``. --frames runs one full ringdown;
iMouse rotates.
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import btz_qnm, btz_temperature
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_RHO_H = wp.constant(0.32)                # the horizon circle
_T_CYCLE = 10.0
_R_H = 0.25                               # bulk horizon radius (sets T through the dictionary)
_L_ADS = 1.0
_K_MODE = 2.0                             # quadrupole kick
_DELTA = 2.0                              # massless scalar: boundary dimension 2
_V_PROP = 0.16                            # outward propagation speed (disk units / s)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, tau: float, om_re: float, om_im: float, kick: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd)
    th = wp.atan2(zd[1], zd[0])
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.0, 0.0, 0.0)

    # the fundamental mode's live amplitude and the retarded field in the bulk
    amp = wp.exp(om_im * tau)

    if r < 1.0:
        col = wp.vec3(0.014, 0.018, 0.048) * (0.55 + 0.45 * (1.0 - r * r))

        # ---- the ringing horizon: a quadrupole-deformed circle, exactly damped ----
        rho = _RHO_H * (1.0 + 0.13 * amp * wp.cos(_K_MODE * th - om_re * tau))
        if r < rho:
            col = wp.vec3(0.010, 0.004, 0.006)
        dh = wp.abs(r - rho)
        wh = wp.max(0.005, 2.0 * px)
        hring = wp.exp(-(dh * dh) / (wh * wh)) + 0.3 * wp.exp(-dh * 16.0)
        col = col + wp.vec3(1.00, 0.42, 0.18) * (hring * (0.55 + 1.3 * amp + 1.8 * kick))

        # ---- the perturbation propagating out at retarded time ----
        if r > rho:
            t_ret = tau - (r - _RHO_H) / _V_PROP
            if t_ret > 0.0:
                psi = wp.exp(om_im * t_ret) * wp.cos(_K_MODE * th - om_re * t_ret)
                fall = wp.exp(-(r - _RHO_H) * 1.1)
                front = wp.clamp(t_ret / 0.5, 0.0, 1.0)   # ease the wavefront in
                col = col + wp.vec3(0.35, 0.70, 1.00) * (psi * psi * fall * front * 0.85)

    # ---- the boundary: flickering with the arriving signal, settling as e^{om_im t} ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    t_bdy = tau - (1.0 - _RHO_H) / _V_PROP
    sig = float(0.0)
    if t_bdy > 0.0:
        sig = (wp.exp(om_im * t_bdy) * wp.cos(_K_MODE * th - om_re * t_bdy)
               * wp.clamp(t_bdy / 0.5, 0.0, 1.0))
    col = col + wp.vec3(0.55, 0.48, 0.36) * ring * 0.7
    col = col + wp.vec3(0.40, 0.75, 1.00) * (ring * sig * sig * 1.4)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # ---- the exact spectrum (fundamental, n = 0), straight from the dictionary ----
    temp = btz_temperature(_R_H, _L_ADS)
    om_re, om_im = btz_qnm(_K_MODE, 0, temp, _DELTA)      # om_im = -4 pi T (Delta/2)
    kick = math.exp(-tau / 0.35)                          # the flash of the initial kick

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(tau), float(om_re), float(om_im), float(kick)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="btz_ringdown",
    description="the only black hole that rings in closed form — a quadrupole kick "
                "excites the BTZ horizon, which rings at exactly omega = k - 4 pi i T "
                "(Delta/2) (Birmingham-Sachs-Solodukhin): the outline oscillates and "
                "damps, cyan ripples carry the perturbation outward along retarded time, "
                "and the boundary ring flickers and settles at the very same rate — "
                "because QNM frequencies ARE the poles of the dual CFT's thermal "
                "correlator: ringdown equals thermalization. --frames runs one kick "
                "and its full decay.",
    renderer=_render,
)
