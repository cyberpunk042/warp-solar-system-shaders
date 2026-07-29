"""Strong lensing by inverse ray shooting — arcs and the Einstein ring, exactly.

The most shader-native physics in the whole engine: for EVERY pixel of the image
plane, run the exact point-lens equation backwards,

    β = θ·(1 − θ_E²/|θ|²),

and ask the source plane what surface brightness lives there (surface brightness is
conserved along rays — Liouville — so this one line IS gravitational lensing; the
engine mirror ``lensing.lens_equation`` is what the suite asserts against the exact
two-image solution ``θ± = (β ± √(β²+4θ_E²))/2``).

A small spiral galaxy drifts behind the lens once per cycle:

* far from alignment it is gently sheared;
* approaching, it stretches into a **tangential arc** outside the Einstein radius
  while a smaller, parity-flipped **counter-image** appears inside (the lens
  equation's second root — always there);
* at near-perfect alignment the pair wraps into the **Einstein ring** — Einstein
  1936, "no hope of observing this phenomenon directly" — now photographed by the
  hundreds (B1938+666, the Cosmic Horseshoe, half the JWST deep fields);
* the faint dashed circle marks θ_E: everything the drama organizes around.

The magnified arcs are *bigger, not brighter per unit area* — lensing conserves
surface brightness while gathering more of it: nature's telescope. --frames runs one
transit; iMouse pans. See ``docs/research/54-gravitational-lensing.md`` (Part I).
"""

import math

import warp as wp

from ..engine import post
from ..scene import Scene

_T_CYCLE = 16.0
_THETA_E = 0.62


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, theta_e: float, sx: float, sy: float, ring_hint: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)

    # ---- faint starfield (unlensed, far foreground) ----
    hsx = wp.sin(x * 73.1 + y * 41.7) * wp.sin(x * 29.3 - y * 67.9)
    if hsx > 0.9975:
        col = col + wp.vec3(0.25, 0.28, 0.35)

    # ---- THE LENS EQUATION, per pixel: beta = theta (1 - thetaE^2/|theta|^2) ----
    r2 = x * x + y * y
    r2s = wp.max(r2, 1.0e-6)
    f = 1.0 - theta_e * theta_e / r2s
    bx = x * f
    by = y * f

    # ---- sample the source galaxy at beta (surface brightness conserved) ----
    dx = bx - sx
    dy = by - sy
    d = wp.sqrt(dx * dx + dy * dy)
    # elliptical spiral: core + two arms + disk falloff
    g = 0.0
    core = wp.exp(-d * d / 0.0035)
    if d < 0.55:
        ph = wp.atan2(dy, dx)
        arm = 0.5 + 0.5 * wp.cos(2.0 * ph - 9.5 * wp.log(wp.max(d, 0.015)))
        disk = wp.exp(-d * d / 0.028)
        g = 1.35 * core + 0.75 * disk * (0.35 + 0.65 * arm * arm)
    gcol = wp.vec3(0.55, 0.75, 1.00) * g + wp.vec3(1.00, 0.80, 0.55) * (1.35 * core)
    col = col + gcol * 0.85

    # ---- the lens: dark mass + warm halo at the center ----
    wl = wp.max(0.045, 3.5 * px)
    col = col * (1.0 - 0.85 * wp.exp(-r2 / (wl * wl)))
    col = col + wp.vec3(0.65, 0.45, 0.25) * (0.5 * wp.exp(-r2 / (wl * wl * 4.0)))

    # ---- the Einstein-radius marker: faint dashed circle ----
    rr = wp.sqrt(r2)
    d_ring = wp.abs(rr - theta_e)
    wr = wp.max(0.004, 1.2 * px)
    dash = 0.5 + 0.5 * wp.sin(wp.atan2(y, x) * 24.0)
    col = col + wp.vec3(0.60, 0.35, 0.95) * ((0.14 + 0.5 * ring_hint) * dash * wp.exp(-(d_ring * d_ring) / (wr * wr)))

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.55, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)
    # the source transits behind the lens: slow ease across, small impact parameter
    s = tau / _T_CYCLE
    sx = -1.55 + 3.10 * s
    sy = 0.045
    # ring hint brightens near perfect alignment
    u = math.sqrt(sx * sx + sy * sy) / _THETA_E
    ring_hint = math.exp(-u * u * 2.2)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(_THETA_E), float(sx), float(sy), float(ring_hint)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="lens_arcs",
    description="strong lensing by exact inverse ray shooting — every pixel runs the "
                "point-lens equation beta = theta(1 - thetaE^2/|theta|^2) backwards "
                "and samples a spiral galaxy drifting behind the lens: gentle shear "
                "becomes a tangential arc plus a parity-flipped counter-image, then "
                "the full Einstein ring at alignment (Einstein 1936: 'no hope of "
                "observing this' — now photographed by the hundreds). Surface "
                "brightness conserved: bigger, not brighter per area. --frames runs "
                "one transit.",
    renderer=_render,
)
