"""The Fermat landscape — lensed images live at stationary points of arrival time.

The deepest way to see a gravitational lens: light takes ALL paths, and images form
where the arrival time is stationary — Fermat's principle in curved spacetime
(Schneider 1985, Blandford & Narayan 1986). For the point lens the arrival-time
surface is exactly

    τ(θ) = |θ − β|²/2 − θ_E²·ln|θ|

(geometric delay: the bent path is longer, plus the Shapiro delay: clocks run slow in
the potential well; ``engine.lensing.fermat_potential_2d``). The scene draws this
landscape live as the source drifts:

* glowing contour lines map the arrival-time surface — a paraboloid dented by the
  logarithmic Shapiro funnel at the lens;
* the **cyan image** sits in the valley (the time MINIMUM — it arrives first), the
  **magenta image** on the mountain pass (the SADDLE — it arrives late): both marked
  at the exact ``θ±``, and the suite asserts ``dτ/dθ = 0`` there numerically, plus
  the saddle's negative tangential curvature;
* the amber ledger tracks the exact delay ``Δτ = τ(θ₋) − τ(θ₊) > 0`` (asserted),
  growing as the source moves off-axis and the landscape tilts.

That delay is measurable: the two images of QSO 0957+561 flicker in the same pattern
417 days apart, and Refsdal 1964 pointed out the lag is proportional to the absolute
size of the universe — time-delay cosmography now measures H₀ to a few percent
(H0LiCOW/TDCOSMO), refereeing the Hubble tension with nothing but geometry and
patience. --frames runs one drift cycle; iMouse pans. See
``docs/research/54-gravitational-lensing.md`` (Part III).
"""

import math

import warp as wp

from ..engine import post
from ..engine.lensing import fermat_potential, image_positions, magnifications, time_delay
from ..scene import Scene

_T_CYCLE = 16.0
_THETA_E = 0.72


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, theta_e: float, bx: float, by: float,
                   ipx: float, ipy: float, ibp: float, imx: float, imy: float, ibm: float,
                   delay_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 3.0
    y = (fy - 0.5 * res[1]) / res[1] * 3.0
    px = 3.0 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- the arrival-time surface tau(theta) ----
    r = wp.sqrt(x * x + y * y)
    rs = wp.max(r, 1.0e-4)
    tau_v = 0.5 * ((x - bx) * (x - bx) + (y - by) * (y - by)) - theta_e * theta_e * wp.log(rs)

    # contour lines: glow at level sets
    lev = tau_v * 3.4
    fr = lev - wp.floor(lev)
    d_line = wp.min(fr, 1.0 - fr)
    depth = wp.exp(-tau_v * 0.55)                      # deeper = warmer base tint
    base = wp.vec3(0.05, 0.08, 0.20) + wp.vec3(0.22, 0.08, 0.02) * depth
    col = col + base * 0.28
    col = col + wp.vec3(0.30, 0.55, 0.95) * (0.42 * wp.exp(-d_line * d_line * 900.0))

    # the Shapiro funnel: darken the infinite-well core
    col = col * (1.0 - 0.75 * wp.exp(-r * r / 0.004))

    # ---- the source's true position (ghost) ----
    d2s = (x - bx) * (x - bx) + (y - by) * (y - by)
    col = col + wp.vec3(0.35, 0.42, 0.55) * (0.22 * wp.exp(-d2s / 0.0035))

    # ---- the images: minimum (cyan, first light) and saddle (magenta, late) ----
    wb = wp.max(0.024, 3.0 * px)
    d2p = (x - ipx) * (x - ipx) + (y - ipy) * (y - ipy)
    col = col + wp.vec3(0.35, 0.90, 1.00) * (ibp * 2.6 * wp.exp(-d2p / (wb * wb)))
    d2m = (x - imx) * (x - imx) + (y - imy) * (y - imy)
    col = col + wp.vec3(1.00, 0.35, 0.80) * (ibm * 2.6 * wp.exp(-d2m / (wb * wb)))

    # ---- the delay ledger (bottom-right, screen frame) ----
    if x > 1.52 and x < 1.60 and y > -1.30 and y < -1.30 + 1.0 * delay_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.1

    uvx = x / 3.0
    uvy = y / 3.0
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.5, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau_c = math.fmod(float(time), _T_CYCLE)
    # the source drifts out and back along a slowly-turning direction
    s = tau_c / _T_CYCLE
    beta_mag = 0.10 + 0.62 * 0.5 * (1.0 - math.cos(2.0 * math.pi * s))
    ang = 0.55 + 0.55 * math.sin(2.0 * math.pi * s * 0.5)
    bx = beta_mag * math.cos(ang)
    by = beta_mag * math.sin(ang)

    tp, tm = image_positions(beta_mag, _THETA_E)
    mp, mm = magnifications(beta_mag, _THETA_E)
    ipx, ipy = tp * math.cos(ang), tp * math.sin(ang)
    imx, imy = tm * math.cos(ang), tm * math.sin(ang)
    ibp = max(min(abs(mp), 4.0) * 0.5, 0.30)
    ibm = max(min(abs(mm), 4.0) * 0.5, 0.30)     # display floor: the saddle image
    # demagnifies to |mu|~0.17 at max offset (physical) but must stay findable
    dly = time_delay(beta_mag, _THETA_E)
    dly_max = time_delay(0.72, _THETA_E)
    _ = fermat_potential(tp, beta_mag, _THETA_E)      # asserted stationary in suite

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(_THETA_E), float(bx), float(by),
                      float(ipx), float(ipy), float(ibp),
                      float(imx), float(imy), float(ibm),
                      float(min(dly / dly_max, 1.0))],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="lens_fermat",
    description="the arrival-time landscape tau = |theta-beta|^2/2 - thetaE^2 ln r "
                "drawn as live contours — a paraboloid dented by the Shapiro funnel "
                "— with the two images at its exact stationary points: cyan in the "
                "valley (the minimum, first light), magenta on the pass (the saddle, "
                "late), both asserted; the amber ledger tracks the exact delay "
                "tau(saddle) - tau(min) > 0 as the source drifts and the landscape "
                "tilts. Refsdal 1964: that lag measures H0 — time-delay cosmography "
                "with nothing but geometry and patience. --frames runs one drift.",
    renderer=_render,
)
