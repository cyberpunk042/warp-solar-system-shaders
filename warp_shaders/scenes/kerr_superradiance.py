"""The black-hole bomb — superradiance, a mirror, and an exponential runaway.

Zel'dovich 1971: shine a wave at a spinning object and, if the wave's angular phase
velocity lags the spin — ``0 < ω < m·Ω_H`` — the scattered wave comes back **amplified**,
carrying away rotational energy. For a Kerr hole this is the wave twin of the Penrose
process (``engine.kerr.superradiant``, the exact condition, test-asserted). Press &
Teukolsky 1972 then noticed the apocalyptic corollary: **wrap the hole in a mirror**.
The amplified wave reflects back, amplifies again, and the amplitude grows as
``A_n = A₀(1 + g)ⁿ`` — exponentially, without limit, until the mirror bursts or the
spin is exhausted. They named it the black-hole bomb. (The same mechanism, with the
mirror replaced by a massive field's potential wall, powers real superradiant
instabilities — the reason LIGO-band holes can constrain ultralight bosons.)

The scene runs one full bomb cycle on the exact dictionary:

* an m = 2 wave (chosen superradiant: ω = ½·m·Ω_H) spirals in from the mirror ring,
  scatters off the ergoregion, and comes back **brighter** — the first amplification;
* trapped between mirror and hole, it re-amplifies every crossing: the spiral bands
  ratchet up as ``(1 + g)ⁿ`` (``bomb_amplitude``, exact) while the horizon's spin
  spokes visibly slow — the wave is mining the same rotational store Penrose does;
* past the critical amplitude the **mirror bursts**: a white flash, the ring shatters
  into fragments that fly off carrying the stored wave energy;
* the wreckage clears on a slower, quieter hole, and the cycle re-arms.

--frames runs one arm-amplify-detonate cycle; iMouse rotates. See
``docs/research/50-kerr-spinning-black-hole.md`` (Part III).
"""

import math

import warp as wp

from ..engine import post
from ..engine.kerr import bomb_amplitude, kerr_horizons, kerr_omega_h, superradiant
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_T_CYCLE = 16.0
_SCALE = wp.constant(0.30)                # world (M = 1) -> disk units
_R_MIRROR = 3.1                           # mirror radius (world)
_M_AZIM = 2
_GAIN = 0.16
_T_PASS = 0.9                             # seconds per mirror round trip
_T_BOOM = 10.6                            # detonation time (amplitude threshold hit)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, tau: float, r_plus: float, amp: float, om_wave: float,
                   spoke_ph: float, mirror_on: float, boom: float, frag_r: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd) / _SCALE
    th = wp.atan2(zd[1], zd[0])
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the trapped superradiant wave: m = 2 spiral standing between hole and mirror ----
    if r > r_plus * 1.05 and r < _R_MIRROR and mirror_on + amp > 0.0:
        env = wp.sin(3.14159265 * (r - r_plus) / (_R_MIRROR - r_plus))
        psi = wp.sin(float(_M_AZIM) * th - om_wave * tau * 6.0 + 3.0 * r) * env
        glow = wp.min(amp * psi * psi * 0.16, 1.6)
        col = col + wp.vec3(0.30, 0.75, 1.00) * glow

    # ---- the mirror: bright ring while armed; shattering fragments after the boom ----
    if mirror_on > 0.0:
        d_m = wp.abs(r - _R_MIRROR) * _SCALE
        wm = wp.max(0.005, 2.0 * px)
        col = col + wp.vec3(0.95, 0.92, 0.85) * (wp.exp(-(d_m * d_m) / (wm * wm)) * 1.3 * mirror_on)
    if frag_r > 0.0:
        k = wp.floor((th + 3.14159265) / 0.5236)         # 12 fragments
        fang = -3.14159265 + (k + 0.5) * 0.5236
        fr = wp.vec2(frag_r * _SCALE * wp.cos(fang), frag_r * _SCALE * wp.sin(fang))
        d2f = (zd[0] - fr[0]) * (zd[0] - fr[0]) + (zd[1] - fr[1]) * (zd[1] - fr[1])
        wf = wp.max(0.010, 3.0 * px)
        col = col + wp.vec3(1.00, 0.85, 0.55) * (2.0 * wp.exp(-d2f / (wf * wf)))

    # ---- the detonation flash ----
    col = col + wp.vec3(1.00, 0.97, 0.90) * (boom * wp.exp(-r * 0.55))

    # ---- the hole: black disk + spin spokes that SLOW as the wave mines the spin ----
    if r < r_plus:
        col = wp.vec3(0.006, 0.003, 0.004)
        spoke = wp.pow(wp.abs(wp.sin(2.0 * th - spoke_ph)), 24.0)
        col = col + wp.vec3(0.85, 0.35, 0.15) * (spoke * wp.clamp(r / r_plus - 0.15, 0.0, 1.0) * 0.8)
    d_h = wp.abs(r - r_plus) * _SCALE
    wh = wp.max(0.005, 2.0 * px)
    col = col + wp.vec3(1.00, 0.42, 0.16) * (wp.exp(-(d_h * d_h) / (wh * wh)) * 1.1)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    m0, a0 = 1.0, 0.90
    r_plus, _ = kerr_horizons(m0, a0)
    om_h = kerr_omega_h(m0, a0)
    om_wave = 0.5 * _M_AZIM * om_h
    assert superradiant(om_wave, _M_AZIM, om_h)      # the condition, live

    # amplitude ratchet: one gain per mirror pass until the boom, then dumped
    if tau < _T_BOOM:
        n = max(int((tau - 1.0) / _T_PASS), 0)
        amp = bomb_amplitude(n, _GAIN)
        mirror_on = 1.0 if tau > 0.4 else tau / 0.4
        boom = 0.0
        frag_r = 0.0
        spin_f = 1.0 - 0.30 * (n / 11.0)             # the wave mines the spin
    else:
        amp = max(0.0, bomb_amplitude(11, _GAIN) * (1.0 - (tau - _T_BOOM) / 0.9))
        mirror_on = 0.0
        boom = math.exp(-(tau - _T_BOOM) / 0.45)
        frag_r = _R_MIRROR + (tau - _T_BOOM) * 2.6
        spin_f = 0.70
    spoke_ph = 5.0 * om_h * spin_f * tau

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(tau), float(r_plus), float(amp), float(om_wave),
                      float(spoke_ph), float(mirror_on), float(boom), float(frag_r)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.5, radius=6)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="kerr_superradiance",
    description="the black-hole bomb — an m=2 wave chosen superradiant (omega < m "
                "Omega_H, the exact Zel'dovich condition) bounces between a spinning "
                "hole and a mirror, amplifying (1+g)^n per pass (Press-Teukolsky): the "
                "cyan spiral ratchets brighter while the horizon's spin spokes slow, "
                "until the mirror BURSTS — white flash, the ring shatters into "
                "fragments, and the wreckage clears on a slower, quieter hole. "
                "--frames runs one arm-amplify-detonate cycle.",
    renderer=_render,
)
