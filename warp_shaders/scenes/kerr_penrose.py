"""The Penrose mine — stealing energy from a black hole until it stops spinning.

Penrose 1969: inside the ergoregion, orbits of NEGATIVE total energy exist. Send a
particle in, let it split; one fragment falls in carrying negative energy, the other
escapes with MORE energy than went in. The books balance because the hole pays: its
mass drops, its spin drops — but Christodoulou's ledger says one thing can never drop:
``M_irr = √(Mr_+/2)``, the area-locked irreducible mass. Mining stops when a = 0 and the
ergoregion closes; the maximum total haul, starting extremal, is
``M(1 − 1/√2) ≈ 29.3%`` of the hole's mass — more efficient than nuclear fusion by
a factor of ~40.

Everything on screen is the exact bookkeeping (``engine.kerr``, test-asserted:
``penrose_extract`` sequences keep M_irr monotone — the area theorem — and the
near-reversible mine recovers 99.9% of the Penrose bound):

* top view: the **horizon disk** (radius r_+) and the **ergoregion annulus** out to the
  stationary limit 2M, its violet swirl rotating at the live ``Ω_H = a/(2Mr_+)``;
* once per cycle a fleet of particles dives in; each **splits at the ergosphere**: the
  crimson fragment spirals through the horizon (negative energy, spin-down), the green
  fragment escapes brighter than it came;
* after every event the dictionary updates: **the mass falls, the spin falls, the swirl
  slows — and the horizon GROWS** (r_+ rises from 1.2M₀ toward 2M_irr): you watch the
  area theorem eat the ergoregion until the annulus pinches shut and the mine is
  exhausted — a bigger, slower, dead hole.

--frames runs one full mining campaign; iMouse rotates. See
``docs/research/50-kerr-spinning-black-hole.md`` (Part II).
"""

import math

import warp as wp

from ..engine import post
from ..engine.kerr import irreducible_mass, kerr_horizons, kerr_omega_h, penrose_extract
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_T_CYCLE = 16.0
_N_EVENTS = 10
_SCALE = wp.constant(0.42)                 # world (M0 = 1) -> disk units


def _schedule():
    """The mining campaign, computed once from the exact bookkeeping."""
    m, a = 1.0, 0.98
    epochs = [(m, a)]
    for _ in range(_N_EVENTS):
        m, a, _ = penrose_extract(m, a, 0.0285, q=0.85)
        epochs.append((m, a))
    return epochs


_EPOCHS = _schedule()


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, r_plus: float, r_ergo: float, om_h: float,
                   swirl_ph: float, px_in: float, py_in: float, pc_in: float,
                   px_g: float, py_g: float, pc_g: float,
                   px_r: float, py_r: float, pc_r: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd) / _SCALE                 # world radius (M0 = 1 units)
    th = wp.atan2(zd[1], zd[0])
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the ergoregion annulus: violet, swirling at the live Omega_H ----
    if r > r_plus and r < r_ergo:
        f_edge = wp.clamp((r_ergo - r) / 0.08, 0.0, 1.0) * wp.clamp((r - r_plus) / 0.05, 0.0, 1.0)
        arms = 0.55 + 0.45 * wp.sin(3.0 * th - swirl_ph - 2.2 * (r - r_plus))
        col = col + wp.vec3(0.34, 0.16, 0.52) * (f_edge * (0.35 + 0.45 * arms))

    # ---- the stationary limit (outer edge of the mine) ----
    d_sl = wp.abs(r - r_ergo) * _SCALE
    wsl = wp.max(0.004, 1.6 * px)
    col = col + wp.vec3(0.62, 0.38, 0.95) * (wp.exp(-(d_sl * d_sl) / (wsl * wsl)) * 0.8)

    # ---- the horizon: black disk, ember rim — WATCH IT GROW ----
    if r < r_plus:
        col = wp.vec3(0.006, 0.003, 0.004)
    d_h = wp.abs(r - r_plus) * _SCALE
    wh = wp.max(0.005, 2.0 * px)
    col = col + wp.vec3(1.00, 0.42, 0.16) * (wp.exp(-(d_h * d_h) / (wh * wh)) * 1.1)

    # ---- the particles: incoming (white), escaping (green, brighter), infalling (red) ----
    wdot = wp.max(0.012, 4.0 * px)
    d2 = (zd[0] - px_in) * (zd[0] - px_in) + (zd[1] - py_in) * (zd[1] - py_in)
    col = col + wp.vec3(0.95, 0.95, 0.90) * (pc_in * 1.8 * wp.exp(-d2 / (wdot * wdot)))
    d2 = (zd[0] - px_g) * (zd[0] - px_g) + (zd[1] - py_g) * (zd[1] - py_g)
    col = col + wp.vec3(0.35, 1.00, 0.45) * (pc_g * 2.6 * wp.exp(-d2 / (wdot * wdot)))
    d2 = (zd[0] - px_r) * (zd[0] - px_r) + (zd[1] - py_r) * (zd[1] - py_r)
    col = col + wp.vec3(1.00, 0.22, 0.16) * (pc_r * 1.6 * wp.exp(-d2 / (wdot * wdot)))

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # epoch: one extraction event every 1.35 s after a 1.2 s establishing shot
    ev = min(max(int((tau - 1.2) / 1.35), 0), _N_EVENTS)
    m, a = _EPOCHS[ev]
    r_plus, _ = kerr_horizons(m, a)
    om_h = kerr_omega_h(m, a) if a > 1e-9 else 0.0
    _ = irreducible_mass(m, a)                    # the monotone ledger (asserted in suite)
    swirl_ph = 14.0 * om_h * tau

    # the active particle triplet (world coords scaled to disk units in-kernel via _SCALE
    # — here we emit screen-disk coordinates directly)
    def polar(rw, ang):
        return (rw * 0.42 * math.cos(ang), rw * 0.42 * math.sin(ang))

    pin = pg = pr = (9.0, 9.0)
    c_in = c_g = c_r = 0.0
    if 1.2 <= tau < 1.2 + _N_EVENTS * 1.35:
        u = ((tau - 1.2) % 1.35) / 1.35
        ang = 0.7 + 2.399 * ev                       # golden-angle spread per event
        r_split = 0.5 * (r_plus + 2.0 * m)           # split mid-ergoregion
        if u < 0.42:                                 # dive
            f = u / 0.42
            pin = polar(2.55 - (2.55 - r_split) * f, ang + 0.25 * f)
            c_in = 1.0
        else:                                        # split: one in, one out, richer
            f = (u - 0.42) / 0.58
            pg = polar(r_split + (2.9 - r_split) * f, ang + 0.25 + 0.55 * f)
            pr = polar(r_split - (r_split - r_plus * 1.02) * f, ang + 0.25 + 1.9 * f)
            c_g = 1.0 - 0.25 * f
            c_r = 1.0 - f * 0.4

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(r_plus), float(2.0 * m),
                      float(om_h), float(swirl_ph),
                      float(pin[0]), float(pin[1]), float(c_in),
                      float(pg[0]), float(pg[1]), float(c_g),
                      float(pr[0]), float(pr[1]), float(c_r)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="kerr_penrose",
    description="the Penrose mine — particles dive into the ergoregion and split: the "
                "crimson fragment falls in with negative energy, the green one escapes "
                "richer, and after each event the exact Christodoulou ledger updates: "
                "mass down, spin down, swirl slower — and the horizon GROWS, because "
                "M_irr never falls (the area theorem, asserted over random extraction "
                "sequences; the near-reversible mine recovers 99.9% of the 29.3% "
                "Penrose bound). The violet annulus pinches shut and the mine dies: a "
                "bigger, slower, dead hole. --frames runs one mining campaign.",
    renderer=_render,
)
