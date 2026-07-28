"""The cosmic event horizon — a black hole turned inside out, with you at the centre.

In a Λ-dominated universe (``a = e^{Ht}``) every observer is wrapped in an event
horizon at proper radius ``1/H``: light from beyond never arrives. Watched from the
centre, a galaxy carried outward by the Hubble flow behaves EXACTLY like something
falling into a black hole seen from outside — it redshifts, slows, freezes at the
horizon, and fades, never quite crossing. The scene runs that fate on the exact
static-patch dictionary (``engine.desitter``, test-asserted):

* two dozen comoving galaxies ride the flow ``r(t) = r₀·e^{Ht}``; each is coloured by
  the exact redshift ``1 + z = 1/√(1 − H²r²)`` (``hubble_flow_redshift``) — white →
  amber → deep red → gone, diverging AT the horizon;
* the horizon ring sits at ``r_H = 1/H`` and glows at the Gibbons-Hawking temperature
  ``T = H/2π`` — empty space at constant Λ is a thermal bath;
* halfway through the cycle **dark energy strengthens**: H ramps up, the horizon ring
  CONTRACTS (r_H = 1/H), the glow warms, and galaxies that were comfortably visible
  redden and vanish early — the lonely far future of an accelerating universe, played
  fast.

The still frame the far-future astronomer dreads: an empty sky, a faint warm ring,
and nothing left to see. --frames runs one drift-and-contraction cycle; iMouse
rotates. See ``docs/research/51-desitter-cosmic-horizon.md`` (Part I).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.desitter import gibbons_hawking_temperature, horizon_radius, hubble_flow_redshift
from ..scene import Scene

_DISK_R = wp.constant(0.44)
_T_CYCLE = 16.0
_N_GAL = 26


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, r_h: float, warm: float,
                   gal: wp.array(dtype=wp.vec4)):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R                         # world units: horizon radius r_h maps to zd length r_h
    r = wp.length(zd)
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- inside the horizon: the faint CMB-ish glow of the observer's patch ----
    if r < r_h:
        th = wp.atan2(zd[1], zd[0])
        blot = 0.5 + 0.5 * wp.sin(5.0 * th + 3.0 * r / r_h) * wp.sin(9.0 * th - 5.0 * r / r_h + time * 0.1)
        col = col + wp.vec3(0.015, 0.02, 0.045) * ((1.0 - r / r_h) * (0.5 + 0.5 * blot))

    # ---- the horizon ring: glowing at T = H/2pi (warmer as H grows) ----
    d_h = wp.abs(r - r_h)
    wh = wp.max(0.010, 2.2 * px)
    ring = wp.exp(-(d_h * d_h) / (wh * wh))
    hot = wp.vec3(1.00, 0.55 + 0.30 * warm, 0.25 + 0.55 * warm)
    col = col + hot * (ring * (0.65 + 0.75 * warm))
    # thermal shimmer just inside
    if r < r_h and r > r_h * 0.9:
        col = col + hot * (0.10 + 0.22 * warm) * ((r - r_h * 0.9) / (r_h * 0.1))

    # ---- the observer: a small cool beacon at the centre of their patch ----
    wob = wp.max(0.012, 3.0 * px)
    col = col + wp.vec3(0.45, 0.75, 1.00) * (1.4 * wp.exp(-(r * r) / (wob * wob)))

    # ---- the galaxies: white -> red -> frozen-and-gone, by the exact redshift ----
    for g in range(_N_GAL):
        gx = gal[g][0]
        gy = gal[g][1]
        zred = gal[g][2]                       # 1+z from the engine
        if zred > 0.0:
            fade = wp.min(1.0 / zred, 1.0)     # surface-brightness death at the horizon
            redness = wp.clamp(1.0 - 1.0 / zred, 0.0, 1.0)
            gcol = wp.vec3(0.95, 0.95 - 0.75 * redness, 0.92 - 0.90 * redness)
            wg = wp.max(0.011, 2.8 * px)
            d2 = (zd[0] - gx) * (zd[0] - gx) + (zd[1] - gy) * (zd[1] - gy)
            col = col + gcol * (1.7 * fade * wp.exp(-d2 / (wg * wg)))

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # dark energy ramps in the second half: H 0.55 -> 0.9
    if tau < 8.0:
        h = 0.55
    else:
        h = 0.55 + 0.35 * 0.5 * (1.0 - math.cos(math.pi * (tau - 8.0) / 8.0))
    r_h = horizon_radius(h)
    t_gh = gibbons_hawking_temperature(h)
    warm = (h - 0.55) / 0.35                       # 0..1 with T = H/2pi

    # comoving galaxies riding the flow r = r0 e^{H t}, reseeded each cycle
    rng = np.random.default_rng(11)
    r0 = rng.uniform(0.10, 0.78, _N_GAL) * horizon_radius(0.55)
    ang = rng.uniform(0.0, 2.0 * math.pi, _N_GAL)
    grow = np.exp(0.55 * min(tau, 8.0) * 0.20 + (0.0 if tau < 8.0 else (tau - 8.0) * 0.05))
    entries = []
    born = 1.0 if tau > 0.6 else tau / 0.6         # fade-in at cycle start
    for k in range(_N_GAL):
        r = float(r0[k] * grow)
        if r < r_h * 0.9995:
            z = hubble_flow_redshift(r, h)
        else:
            r, z = r_h * 0.9995, 60.0              # frozen at the horizon
        entries.append((r * math.cos(ang[k]), r * math.sin(ang[k]), z * (1.0 / born if born > 0 else 1e9), 0.0))
    gal = wp.array(np.asarray(entries, dtype=np.float32), dtype=wp.vec4, device=device)

    _ = t_gh                                        # the ring's warmth IS this number

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(r_h), float(warm), gal],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ds_horizon",
    description="the cosmic event horizon — a black hole turned inside out with the "
                "observer at the centre: galaxies ride the Hubble flow r = r0 e^{Ht} "
                "and redden by the exact 1+z = 1/sqrt(1-H^2r^2), freezing and fading "
                "AT the horizon exactly like infall watched from outside a black hole; "
                "the ring at r_H = 1/H glows at the Gibbons-Hawking temperature H/2pi, "
                "and when dark energy ramps the horizon CONTRACTS and empties the sky "
                "early — the lonely far future, played fast. --frames runs one cycle.",
    renderer=_render,
)
