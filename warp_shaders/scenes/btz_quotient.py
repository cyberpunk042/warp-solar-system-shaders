"""The BTZ black hole IS a quotient of the Poincaré disk.

In 2+1 dimensions gravity has no local degrees of freedom — every solution is locally
pure AdS₃. And yet there is a black hole (Bañados-Teitelboim-Zanelli 1992), because a
black hole here is not local curvature: it is **global identification**. Take the
Poincaré-disk time slice of AdS₃ and quotient by ONE hyperbolic isometry — a translation
by hyperbolic distance λ along a geodesic axis — and you get the BTZ hole:

* the **axis projects to the horizon**, a closed geodesic of length λ = 2πr_h/L;
* the **entropy is that length**, S = λL/4G = 2πr_h/4G — the quantity Strominger
  counted exactly with the boundary CFT's Cardy formula;
* the **temperature is linear in the horizon**, T = r_h/(2πL²) (host-side
  ``btz_temperature`` — always positive specific heat: the 3D box is a perfect cavity).

The scene draws the quotient construction itself, live on the disk:

* the glowing **horizon geodesic** — the horizontal diameter, the axis of the
  identification, brightness rising with T;
* the **fundamental-domain walls**: geodesics orthogonal to the axis crossing it at
  ``x_n = tanh(n·λ/2)`` (host-side ``quotient_wall_position``) — the images of one wall
  under powers of the generator, visibly **accumulating at the two fixed points ±1** on
  the boundary. Everything between two adjacent walls is one copy of the whole black
  hole exterior; the tinted strip is the fundamental domain;
* the horizon **breathes** once per cycle: r_h sweeps up and down, λ = 2πr_h/L with it,
  and you watch the walls spread apart as the hole grows — entropy as geometry, live.

Every drawn element is the honest hyperbolic construction (orthocircle geodesics on the
disk, exact wall positions); see ``docs/research/48-btz-black-hole-on-the-disk.md``.
--frames runs one breathing cycle; iMouse rotates.
"""

import math

import warp as wp

from ..engine import post
from ..engine.adscft import (
    btz_entropy,
    btz_temperature,
    horizon_translation_length,
    quotient_wall_position,
)
from ..scene import Scene

_DISK_R = wp.constant(0.43)
_L_ADS = 1.0
_T_CYCLE = 12.0
_N_WALLS = 6                              # walls drawn each side (accumulate at ±1)


@wp.func
def _wall_glow(zd: wp.vec2, xn: float, px: float) -> float:
    """Glow of the geodesic orthogonal to the x-axis crossing it at (xn, 0): the
    orthocircle centred (d, 0), d = (1 + xn²)/(2xn), rad² = d² − 1 (a diameter when
    xn ~ 0)."""
    d = float(0.0)
    if wp.abs(xn) < 1.0e-3:
        d = wp.abs(zd[0])                 # the seed wall: the vertical diameter
    else:
        cx = (1.0 + xn * xn) / (2.0 * xn)
        rad = wp.sqrt(wp.max(cx * cx - 1.0, 1.0e-10))
        d = wp.abs(wp.length(zd - wp.vec2(cx, 0.0)) - rad)
    w = wp.max(0.004, 1.6 * px)
    return wp.exp(-(d * d) / (w * w)) + 0.30 * wp.exp(-d * 22.0)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, lam: float, t_norm: float,
                   walls: wp.array(dtype=float), n_walls: int):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])

    zd = uv / _DISK_R
    r = wp.length(zd)
    px = 1.0 / (_DISK_R * res[1])

    col = wp.vec3(0.0, 0.0, 0.0)

    if r < 1.0:
        # ---- the bulk: a quiet hyperbolic gradient (locally pure AdS3 everywhere) ----
        col = wp.vec3(0.014, 0.020, 0.050) * (0.55 + 0.45 * (1.0 - r * r))

        # ---- the horizon: the axis geodesic, a closed curve of length lambda ----
        dh = wp.abs(zd[1])
        wh = wp.max(0.005, 2.0 * px)
        hglow = wp.exp(-(dh * dh) / (wh * wh)) + 0.35 * wp.exp(-dh * 18.0)
        col = col + wp.vec3(1.00, 0.30, 0.16) * (hglow * (0.55 + 1.1 * t_norm))

        # ---- the identification walls, accumulating at the fixed points ----
        for n in range(2 * n_walls + 1):
            g = _wall_glow(zd, walls[n], px)
            fade = 1.0 - 0.09 * wp.abs(float(n - n_walls))
            col = col + wp.vec3(1.00, 0.74, 0.30) * (g * 0.85 * fade)

        # ---- the fundamental domain: one copy of the entire exterior ----
        x1 = walls[n_walls + 1]
        cx = (1.0 + x1 * x1) / (2.0 * wp.max(x1, 1.0e-4))
        rad = wp.sqrt(wp.max(cx * cx - 1.0, 1.0e-10))
        inside_dom = float(0.0)
        if zd[0] > 0.0 and wp.length(zd - wp.vec2(cx, 0.0)) > rad:
            inside_dom = 1.0
        col = col + wp.vec3(0.30, 0.16, 0.42) * (inside_dom * 0.22)

    # ---- the conformal boundary + the two fixed points of the isometry ----
    bw = wp.max(0.004, 1.8 * px)
    ring = wp.exp(-((r - 1.0) * (r - 1.0)) / (bw * bw)) + 0.25 * wp.exp(-wp.abs(r - 1.0) * 16.0)
    col = col + wp.vec3(0.55, 0.48, 0.36) * ring * 0.7
    dfix = wp.min(wp.length(zd - wp.vec2(1.0, 0.0)), wp.length(zd + wp.vec2(1.0, 0.0)))
    col = col + wp.vec3(1.00, 0.95, 0.80) * (2.0 * wp.exp(-(dfix * dfix) / (30.0 * bw * bw)))

    col = col * (1.0 - 0.35 * wp.min(wp.length(uv) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)

    # ---- the hole breathes: r_h sweeps, and lambda / T / S follow the dictionary ----
    r_h = 0.19 + 0.11 * math.sin(2.0 * math.pi * tau / _T_CYCLE - 0.5 * math.pi)
    lam = horizon_translation_length(r_h, _L_ADS)          # = 2 pi r_h / L
    temp = btz_temperature(r_h, _L_ADS)
    t_norm = temp / btz_temperature(0.3, _L_ADS)           # vs the sweep's hottest hole
    _ = btz_entropy(r_h)                                   # = lam * L / 4: entropy IS length

    xs = [quotient_wall_position(n, lam) for n in range(-_N_WALLS, _N_WALLS + 1)]
    walls = wp.array(xs, dtype=float, device=device)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(lam), float(t_norm), walls, _N_WALLS],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.45, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.12, preserve_hue=True)


SCENE = Scene(
    name="btz_quotient",
    description="the BTZ black hole built before your eyes — 3D gravity has no local "
                "curvature, so its black hole is a global identification: the Poincare "
                "disk quotiented by one hyperbolic isometry. The axis geodesic glows as "
                "the horizon (a closed curve of length 2 pi r_h/L = the entropy), the "
                "fundamental-domain walls march along at x_n = tanh(n lambda/2) and "
                "accumulate at the isometry's two boundary fixed points, and the hole "
                "breathes once per cycle so you watch entropy-as-length live. "
                "--frames runs one breathing cycle.",
    renderer=_render,
)
