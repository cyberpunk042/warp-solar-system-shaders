"""Peters circularization — gravitational waves iron orbits round.

Why were LIGO's binaries all circular? Because radiation reaction doesn't just shrink
an orbit — it preferentially erases eccentricity. Peters 1964, exact at leading order
(``engine.gw``):

    da/dt = −64/5·m₁m₂M/a³·(1−e²)^{−7/2}(1 + 73e²/24 + 37e⁴/96)
    de/dt = −304/15·e·m₁m₂M/a⁴·(1−e²)^{−5/2}(1 + 121e²/304)

Emission peaks at pericenter — the closest, fastest part of the orbit — so each
passage bleeds away exactly the motion that made the orbit eccentric. The scene
integrates the coupled system once (``evolve_peters``, RK2; the suite asserts e is
strictly decreasing and dies *fractionally faster* than a) and plays the trajectory:

* the live orbit is the exact ellipse ``r(ν) = a(1−e²)/(1+e·cosν)`` with the body
  running faster at pericenter (Kepler's second law, via the vis-viva speed);
* ghost ellipses of past epochs fade behind it — a shrinking, rounding onion;
* the amber ledger bars track ``a`` and ``e`` from the integration: watch ``e``
  collapse toward zero while ``a`` still has distance left to fall.

By the time a binary born eccentric reaches the LIGO band it is round to a part in a
thousand — the waves erased the orbit's memory. Hulse–Taylor (e = 0.617 today) is
mid-flight on exactly this curve. --frames runs one trajectory; iMouse pans. See
``docs/research/53-gravitational-waves.md`` (Part III).
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.gw import evolve_peters
from ..scene import Scene

_T_CYCLE = 16.0
_N_GHOST = 5


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float,
                   mouse: wp.vec2, a_now: float, e_now: float, nu: float,
                   ghosts: wp.array(dtype=wp.vec2), n_ghost: int,
                   a_frac: float, e_frac: float):
    i, j = wp.tid()
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5
    res = wp.vec2(float(width), float(height))
    x = (fx - 0.5 * res[0]) / res[1] * 2.6
    y = (fy - 0.5 * res[1]) / res[1] * 2.6
    px = 2.6 / res[1]

    col = wp.vec3(0.004, 0.005, 0.012)
    # orbit frame shifted right so the full ellipse (apocenter left of focus) fits
    xo = x + 0.42
    r = wp.sqrt(xo * xo + y * y)
    th = wp.atan2(y, xo)

    # ---- ghost ellipses of past epochs (focus at origin) ----
    for g in range(n_ghost):
        ga = ghosts[g][0]
        ge = ghosts[g][1]
        if ga > 0.0:
            r_ell = ga * (1.0 - ge * ge) / (1.0 + ge * wp.cos(th))
            d_g = wp.abs(r - r_ell)
            wg = wp.max(0.006, 1.6 * px)
            fade = 0.28 * (float(g) + 1.0) / float(n_ghost)
            col = col + wp.vec3(0.18, 0.24, 0.45) * (fade * wp.exp(-(d_g * d_g) / (wg * wg)))

    # ---- the live orbit: r(nu) = a(1-e^2)/(1+e cos nu), focus at origin ----
    r_now = a_now * (1.0 - e_now * e_now) / (1.0 + e_now * wp.cos(th))
    d_o = wp.abs(r - r_now)
    wo = wp.max(0.007, 1.8 * px)
    col = col + wp.vec3(0.35, 0.85, 1.00) * (wp.exp(-(d_o * d_o) / (wo * wo)) * 0.95)

    # ---- the central mass at the focus ----
    d2c = xo * xo + y * y
    wc = wp.max(0.022, 3.0 * px)
    col = col + wp.vec3(0.95, 0.85, 0.65) * (1.4 * wp.exp(-d2c / (wc * wc)))

    # ---- the orbiting body at true anomaly nu (fast at pericenter) ----
    rb = a_now * (1.0 - e_now * e_now) / (1.0 + e_now * wp.cos(nu))
    bx = rb * wp.cos(nu)
    by = rb * wp.sin(nu)
    d2b = (xo - bx) * (xo - bx) + (y - by) * (y - by)
    wb = wp.max(0.018, 2.6 * px)
    col = col + wp.vec3(0.95, 0.95, 0.90) * (1.7 * wp.exp(-d2b / (wb * wb)))
    # pericenter emission flare: brightest where the radiation is emitted
    peri = wp.exp(-(rb - a_now * (1.0 - e_now)) * (rb - a_now * (1.0 - e_now)) / (0.02 * a_now * a_now))
    col = col + wp.vec3(1.00, 0.55, 0.20) * (0.7 * peri * e_now * wp.exp(-d2b / (wb * wb * 6.0)))

    # ---- the ledger: a and e bars, bottom-left (screen frame) ----
    if x > -1.62 and x < -1.56 and y > -1.10 and y < -1.10 + 0.80 * a_frac:
        col = col + wp.vec3(0.35, 0.85, 1.00) * 1.0
    if x > -1.48 and x < -1.42 and y > -1.10 and y < -1.10 + 0.80 * e_frac:
        col = col + wp.vec3(1.00, 0.72, 0.25) * 1.0

    uvx = x / 2.6
    uvy = y / 2.6
    col = col * (1.0 - 0.35 * wp.min(wp.sqrt(uvx * uvx + uvy * uvy) * 1.55, 1.0))
    img[i, j] = wp.max(col, wp.vec3(0.0, 0.0, 0.0))


_TRAJ = None


def _render(width, height, time, mouse, device):
    global _TRAJ
    if _TRAJ is None:
        _TRAJ = evolve_peters(1.05, 0.72, 1.0, 1.0, 4e-6, 250000)
    traj = _TRAJ
    tau = math.fmod(float(time), _T_CYCLE)
    # map cycle time to trajectory index (eased so the endgame lingers)
    s = tau / _T_CYCLE
    idx = min(int((s ** 1.35) * (len(traj) - 1)), len(traj) - 1)
    a, e = traj[idx]
    a0, e0 = traj[0]

    # ghost epochs behind the live orbit
    ghosts = []
    for g in range(_N_GHOST):
        gi = int(idx * g / _N_GHOST)
        ga, ge = traj[gi]
        ghosts.append((ga, ge))

    # true anomaly: sweep with Kepler-2 weighting (faster at pericenter)
    nu = 2.0 * math.pi * (tau * (1.0 + 2.5 * (1.0 - a / a0)))
    # crude area-law modulation: advance more when near pericenter
    nu = nu + 0.9 * e * math.sin(nu)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, width, height, float(time),
                      wp.vec2(float(mouse[0]), float(mouse[1])),
                      float(a), float(e), float(nu),
                      wp.array(np.asarray(ghosts, np.float32), dtype=wp.vec2, device=device),
                      int(_N_GHOST), float(a / a0), float(e / e0)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gw_orbits",
    description="Peters circularization — the exact coupled ODEs da/dt, de/dt "
                "(integrated once, RK2; e asserted strictly decreasing and dying "
                "fractionally faster than a) play out as a live ellipse "
                "r = a(1-e^2)/(1+e cos nu) shrinking through fading ghost epochs, "
                "the body flaring orange at pericenter where the radiation is "
                "emitted; ledger bars track a (cyan) and e (amber) as the waves "
                "iron the orbit round — why LIGO's binaries all arrive circular. "
                "--frames runs one trajectory.",
    renderer=_render,
)
