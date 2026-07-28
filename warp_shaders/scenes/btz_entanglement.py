"""The entanglement plateau — when your interval swallows the black hole.

Finite-temperature Ryu-Takayanagi, on the BTZ background. A boundary interval of size θ
now has TWO candidate minimal surfaces, because the homology constraint counts the
horizon:

* **direct** — the geodesic hugging the interval: ``S = S_th(θ)``, the finite-T
  Calabrese-Cardy entropy ``(c/3)·ln((β/πε)·sinh(πθ/β))`` (host-side
  ``thermal_interval_entropy``): the vacuum log for θ ≪ β, EXTENSIVE with slope
  ``(πc/3)T`` for θ ≫ β — the interval stops learning geometry and just counts thermal
  excitations;
* **wrapped** — the *complement's* geodesic PLUS the horizon itself, which homology
  forces the surface to wrap: ``S = S_th(2π − θ) + S_BH``.

RT takes the MINIMUM (``thermal_entanglement``) — the fourth saddle competition of the
set, after mutual information, Hawking-Page, and the Page curve's island. The swap at
the **plateau angle** θ* (host-side ``plateau_angle``, bisected, test-asserted) is
Hubeny-Maxfield-Rangamani-Tonni's entanglement plateau: past θ*, S(θ) freezes at the
wrapped value, and Araki-Lieb ``|S(A) − S(Ā)| ≤ S_BH`` is EXACTLY saturated — the
interval's entanglement wedge now contains the entire black hole.

The scene sweeps θ through θ* once per cycle:

* the magenta interval grows along the boundary; its **direct geodesic** hangs bright
  while it wins;
* at θ* the drawing swaps: the direct arc ghosts out, the **complement's arc** and the
  **horizon circle** ignite violet — the surface now wraps the hole — and the
  entanglement-wedge tint floods everything outside the complement's little cap: the
  black hole *belongs to the interval*;
* sweep back, the plateau releases, the hole escapes the wedge.

The arcs are pure-AdS orthocircles (schematic in the BTZ metric — flagged); the
entropies, the plateau angle and the saturation are the honest closed forms. See
``docs/research/48-btz-black-hole-on-the-disk.md``. --frames runs one sweep; iMouse
rotates.
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import (
    geodesic_far_side,
    in_boundary_arc,
    plateau_angle,
    rt_geodesic_glow,
    thermal_entanglement,
    thermal_interval_entropy,
)
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_RHO_H = wp.constant(0.30)                # the horizon circle (schematic embedding)
_T_CYCLE = 14.0
_TEMP = 0.35
_S_BH = 1.5
_EPS_UV = 0.05


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, a1: float, a2: float, wrapped: float):
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

    if r < 1.0:
        # ---- the thermal bulk, and the horizon sitting in the middle of it ----
        col = wp.vec3(0.016, 0.018, 0.046) * (0.55 + 0.45 * (1.0 - r * r))
        if r < _RHO_H:
            col = wp.vec3(0.010, 0.004, 0.006)            # inside the hole: dark
        dh = wp.abs(r - _RHO_H)
        wh = wp.max(0.005, 2.0 * px)
        hring = wp.exp(-(dh * dh) / (wh * wh)) + 0.3 * wp.exp(-dh * 18.0)
        # the horizon ring: dim ember while unclaimed, blazing violet once WRAPPED
        # (homology: the RT surface now includes it)
        ember = wp.vec3(0.85, 0.28, 0.14) * (0.35 * (1.0 - wrapped))
        claimed = wp.vec3(0.75, 0.40, 1.00) * (1.5 * wrapped)
        col = col + (ember + claimed) * hring

        # ---- the entanglement wedge ----
        if r > _RHO_H:
            # direct phase: the cap between the interval and its geodesic
            w_dir = 1.0 - geodesic_far_side(zd, a1, a2)
            # wrapped phase: EVERYTHING except the complement's little cap
            w_wrap = geodesic_far_side(zd, a2, a1)
            wedge = wp.lerp(w_dir, w_wrap, wrapped)
            col = col + wp.vec3(0.38, 0.18, 0.55) * (wedge * 0.34)

    # ---- the conformal boundary + the interval A living on it ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    col = col + wp.vec3(0.55, 0.48, 0.36) * ring * 0.7
    band = wp.exp(-((r - 1.0) * (r - 1.0)) / (9.0 * bw * bw))
    col = col + wp.vec3(1.00, 0.30, 0.78) * (band * in_boundary_arc(th, a1, a2) * 1.6)

    # ---- the two competing RT candidates: winner bright, loser ghost-faint ----
    inside = wp.clamp((1.03 - r) / 0.03, 0.0, 1.0)
    outside_hole = wp.clamp((r - _RHO_H - 0.01) / 0.02, 0.0, 1.0)
    g_dir = rt_geodesic_glow(zd, a1, a2, px) * outside_hole
    g_wrap = rt_geodesic_glow(zd, a2, a1, px) * outside_hole
    amp_d = wp.lerp(1.15, 0.16, wrapped) * inside
    amp_w = wp.lerp(0.16, 1.15, wrapped) * inside
    col = col + wp.vec3(1.00, 0.62, 0.30) * (g_dir * amp_d)
    col = col + wp.vec3(0.90, 0.75, 1.00) * (g_wrap * amp_w)

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # the interval sweeps up through the plateau angle and back, once per cycle
    theta = 0.55 + (2.0 * math.pi - 1.15) * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))
    s_now, wrapped_now = thermal_entanglement(theta, _TEMP, _S_BH, _EPS_UV)
    th_star = plateau_angle(_TEMP, _S_BH, _EPS_UV)
    # smooth the visual swap around the exact plateau angle
    wrapped = 0.5 * (1.0 + math.tanh((theta - th_star) / 0.12))
    _ = (s_now, wrapped_now, thermal_interval_entropy(theta, _TEMP, _EPS_UV))

    # interval centred on the top of the disk
    a1 = 0.5 * math.pi - 0.5 * theta
    a2 = 0.5 * math.pi + 0.5 * theta

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a1), float(a2), float(wrapped)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="btz_entanglement",
    description="the entanglement plateau at finite temperature — a boundary interval on "
                "the BTZ background sweeps through the plateau angle where RT's two "
                "candidates swap: the direct geodesic (finite-T Calabrese-Cardy entropy, "
                "extensive for theta >> beta) loses to the complement's geodesic PLUS the "
                "horizon, which homology forces the surface to wrap — the horizon ring "
                "ignites violet, the wedge floods the disk, and Araki-Lieb saturates "
                "exactly: the interval's wedge now contains the entire black hole. "
                "--frames runs one sweep through the plateau.",
    renderer=_render,
)
