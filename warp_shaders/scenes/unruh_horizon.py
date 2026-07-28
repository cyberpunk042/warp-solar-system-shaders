"""Unruh — accelerate hard enough and empty space glows.

The third member of the temperature trilogy, and the one that gives the game away:
Hawking's ``T = κ/2π`` and Gibbons-Hawking's ``T = H/2π`` were never about black holes
or cosmology — a horizon plus quantum fields equals heat, and acceleration alone makes
a horizon. Unruh 1976: an observer with proper acceleration ``a`` measures the
Minkowski vacuum as a thermal bath at

    T = a/2π

(``engine.vacuum.unruh_temperature``; the suite asserts the trilogy numerically
against ``engine.desitter``). The scene is a live spacetime diagram:

* the observer's worldline is the exact hyperbola ``x² − t² = 1/a²``
  (``rindler_worldline``, on-hyperbola identity asserted), riding it as proper time
  advances;
* the **Rindler horizon** — the null line x = t — glows violet: everything behind it
  is unseeable *forever*, a horizon with no black hole anywhere; the unknowable wedge
  is shaded out;
* the horizon trails the observer at proper distance ``1/a``
  (``rindler_horizon_distance``): once per cycle the acceleration ramps 0.25 → 2.5,
  the hyperbola tightens into the corner, the private horizon closes in — and the
  **thermal bath around the observer brightens and warms** exactly as a/2π.

Accelerate, and the vacuum answers with heat. --frames runs one ramp of a; iMouse
pans. See ``docs/research/52-quantum-vacuum.md`` (Part I).
"""

import math

import warp as wp

from ..engine import post
from ..engine.vacuum import rindler_horizon_distance, rindler_worldline, unruh_temperature
from ..scene import Scene

_T_CYCLE = 16.0


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, a_acc: float, obs_x: float, obs_t: float, warm: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    # diagram coordinates: x in [-0.25, 2.4], t in [-1.1, 1.1]
    x = -0.25 + (fx / res[0]) * 2.65
    t = -1.1 + (fy / res[1]) * 2.2
    px = 2.65 / res[0]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- light-cone grid: faint 45-degree lines ----
    gu = wp.abs(wp.sin((x + t) * 7.854))
    gv = wp.abs(wp.sin((x - t) * 7.854))
    col = col + wp.vec3(0.012, 0.015, 0.026) * (wp.exp(-gu * 16.0) + wp.exp(-gv * 16.0))

    # ---- the unknowable wedge (behind the horizon): shaded out ----
    if x < wp.abs(t):
        col = col * 0.45 + wp.vec3(0.012, 0.006, 0.020)

    # ---- the Rindler horizon: the null line x = t (and its past mirror, fainter) ----
    d_hp = wp.abs(x - t) * 0.7071
    wh = wp.max(0.006, 1.8 * px)
    col = col + wp.vec3(0.62, 0.30, 1.00) * (wp.exp(-(d_hp * d_hp) / (wh * wh)) * 1.1)
    d_hm = wp.abs(x + t) * 0.7071
    col = col + wp.vec3(0.40, 0.20, 0.65) * (wp.exp(-(d_hm * d_hm) / (wh * wh)) * 0.45)

    # ---- the worldline: the exact hyperbola x^2 - t^2 = 1/a^2 (right wedge) ----
    if x > 0.02:
        f = x * x - t * t - 1.0 / (a_acc * a_acc)
        # gradient magnitude ~ 2*sqrt(x^2+t^2): normalize for even stroke width
        grad = 2.0 * wp.sqrt(x * x + t * t)
        d_w = wp.abs(f) / wp.max(grad, 0.2)
        ww = wp.max(0.006, 1.8 * px)
        col = col + wp.vec3(0.35, 0.85, 1.00) * (wp.exp(-(d_w * d_w) / (ww * ww)) * 0.9)

    # ---- the observer riding it, wrapped in their thermal bath at T = a/2pi ----
    d2 = (x - obs_x) * (x - obs_x) + (t - obs_t) * (t - obs_t)
    wob = wp.max(0.012, 3.0 * px)
    col = col + wp.vec3(0.95, 0.95, 0.90) * (1.6 * wp.exp(-d2 / (wob * wob)))
    # the bath: warm speckled glow, radius and heat growing with the temperature
    bath_r = 0.05 + 0.22 * warm
    speck = 0.6 + 0.4 * wp.sin(x * 55.0 + time * 3.0) * wp.sin(t * 47.0 - time * 2.2)
    hot = wp.vec3(1.00, 0.45 + 0.30 * warm, 0.18 + 0.30 * warm)
    col = col + hot * (warm * 1.05 * speck * wp.exp(-d2 / (bath_r * bath_r)))

    uvx = fx / res[1] - 0.5 * res[0] / res[1]
    uvy = fy / res[1] - 0.5
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.1, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau_c = math.fmod(float(time), _T_CYCLE)

    # acceleration ramps up and back once per cycle
    a = 0.25 + 2.25 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau_c / _T_CYCLE))
    t_u = unruh_temperature(a)
    warm = (a - 0.25) / 2.25
    _ = rindler_horizon_distance(a)               # the hyperbola vertex sits at exactly this x

    # ride the worldline: proper time oscillates so the observer sweeps up and down
    tau_obs = 0.9 * math.sin(2.0 * math.pi * tau_c / 4.0) / max(a, 0.4)
    ox, ot = rindler_worldline(a, tau_obs)
    _ = t_u

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a), float(ox), float(ot), float(warm)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="unruh_horizon",
    description="the Unruh effect on a live spacetime diagram — the observer rides "
                "the exact hyperbola x^2 - t^2 = 1/a^2 while the violet Rindler "
                "horizon (the null line x = t, a horizon with no black hole anywhere) "
                "trails at proper distance 1/a; as the acceleration ramps, the "
                "hyperbola tightens into the corner, the private horizon closes in, "
                "and the thermal bath around the observer brightens at exactly "
                "T = a/2pi — the flat-space member of the kappa/2pi, H/2pi, a/2pi "
                "temperature trilogy. --frames runs one ramp.",
    renderer=_render,
)
