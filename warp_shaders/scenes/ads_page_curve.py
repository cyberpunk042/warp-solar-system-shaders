"""The Page curve — an evaporating black hole kept honest by the island saddle.

Hawking's 1974 calculation says the radiation of an evaporating hole is thermal, its
entropy rising monotonically until the hole is gone — and then the information that fell
in is simply lost. Page (1993) said unitarity demands otherwise: the radiation entropy
must **turn over** when it equals the hole's remaining Bekenstein-Hawking entropy, and
return to zero when evaporation completes. The 2019 resolution (Penington;
Almheiri-Engelhardt-Marolf-Maxfield) found Page's curve *inside gravity itself*: the
gravitational path integral computes the radiation entropy as a MINIMUM over saddles —

* **Hawking saddle** (no island): ``S = S₀ − S_BH(t)``, rising as radiation accumulates;
* **island saddle**: a quantum extremal surface just inside the horizon hands the entire
  interior to the radiation's entanglement wedge, at the cost of the horizon area:
  ``S = S_BH(t)``, falling as the hole shrinks.

The same min-over-saddles rule as ``ads_entanglement``'s pairing swap and
``ads_hawking_page``'s ensemble dominance — the whole set is one theme, and this is its
capstone. The dictionary is closed-form in the engine core (``bh_entropy_evaporating``,
``page_curve``, ``page_time`` — the crossing at ``t_page = T(1 − 2^{−3/2}) ≈ 0.646 T``,
test-asserted).

The scene runs one full evaporation per cycle, in the AdS box:

* The hole shrinks on the honest Stefan-Boltzmann schedule ``r_h ∝ (1 − t/T)^{1/3}`` —
  slow at first, then the runaway; its Hawking temperature **rises** as it shrinks
  (small holes are hot), flashing at the end.
* Its Hawking radiation glows as an aura hugging the horizon (the Hartle-Hawking bath),
  brightening with the Hawking-saddle entropy — before the Page time it is featureless
  and thermal (orange: information-free).
* At the **Page time** the island saddle wins: a violet **quantum extremal surface**
  ignites just outside the horizon (the island's edge — the interior now belongs to the
  radiation), and the radiation *purifies* — cooling from thermal orange to coherent
  cyan-white as its entropy comes back down. Unitarity, watched in real time.
* At ``t = T`` the hole is gone, the box holds pure structured light, and the cycle
  (like the GIF) begins again.

See ``docs/research/46-ads-cft-holography.md`` (Part VII). --frames runs one full
evaporation; iMouse orbits.
"""

import math

import warp as wp

from .. import lod
from ..engine import post
from ..engine.adscft import (
    bh_entropy_evaporating,
    boundary_cft,
    hawking_page_temperature,
    hawking_temperature,
    horizon_radius,
    mass_of_radius,
    page_curve,
    page_time,
)
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_L_ADS = 2.2
_R_BDY = wp.constant(24.0)
_T_EVAP = 16.0                            # one full evaporation per cycle (seconds)
_R_H0 = 2.86                              # initial horizon (1.3 L — a large, stable hole)
_BOUNCES = {"low": 1, "medium": 2, "high": 3, "ultra": 4}


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float, max_steps: int, max_bounce: int,
                   m_bh: float, r_cap: float, t_hawk: float, s_hawk: float,
                   island: float, qes_r: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanf * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanf
    rd = wp.normalize(fwd + right * u + up * v)

    pos = eye
    vel = rd
    # per-pixel phase jitter on the march start: decorrelates the aura's step-sampling
    # rings into noise (the aura is a smooth shell sampled at discrete dt)
    jit = wp.sin(float(i * 1973 + j * 9277) * 0.7391) * 43758.547
    jit = jit - wp.floor(jit)
    pos = pos + vel * (0.30 * jit)
    cr = wp.cross(pos, vel)
    h2 = wp.dot(cr, cr)

    col = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    bounce = int(0)
    gas = float(0.0)                       # the accumulated Hawking radiation
    qes = float(0.0)                       # the quantum extremal surface shell

    for _s in range(max_steps):
        r = wp.length(pos)
        if r_cap > 0.0 and r < r_cap:      # captured — the (shrinking) shadow
            break

        if r > _R_BDY:                     # reflecting conformal boundary
            n = pos / r
            col = col + boundary_cft(n, time, t_hawk) * (trans * 0.38)
            bounce += 1
            if bounce > max_bounce:
                break
            trans = trans * 0.42
            vel = vel - n * (2.0 * wp.dot(vel, n))
            pos = n * (_R_BDY - 1.0e-3)
            cr2 = wp.cross(pos, vel)
            h2 = wp.dot(cr2, cr2)

        # exact Schwarzschild-AdS photon pull (Lambda drops out of the path shape)
        acc = pos * (-3.0 * m_bh * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.5)
        vel = vel + acc * dt
        pos = pos + vel * dt

        # the Hawking radiation: the Hartle-Hawking aura hugging the (shrinking) horizon,
        # growing with the emitted-so-far Hawking-saddle entropy
        if r_cap > 0.0:
            gas += dt * wp.exp(-(r - r_cap) * 1.6) * trans
        else:
            gas += dt * wp.exp(-r * 0.5) * trans          # the last burst, hole gone
        # the island's edge: a crisp QES ring just outside the horizon, alive after t_page
        if island > 0.001 and r_cap > 0.0:
            dq = (r - qes_r) / 0.12
            qes += dt * wp.exp(-dq * dq) * trans

    # the radiation follows the Page story: amount grows with the Hawking saddle, the
    # colour purifies (thermal orange -> coherent cyan) once the island dominates — and
    # captured rays carry the glow INTO the capture disk, so before the Page time the
    # shadow is near-black (information going in, nothing coming back) while after it
    # the interior itself lights up: the island now belongs to the radiation's wedge
    rad_c = wp.lerp(wp.vec3(1.00, 0.50, 0.20), wp.vec3(0.55, 0.90, 1.00), island)
    col = col + rad_c * (gas * (0.03 + 0.30 * s_hawk * island))
    col = col + wp.vec3(0.72, 0.35, 1.00) * (qes * island * 1.2)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = lod.active_tier()
    max_steps = tier.raymarch_steps * 4
    max_bounce = _BOUNCES.get(tier.name, 2)

    tau = math.fmod(float(time), _T_EVAP)

    # ---- the evaporation schedule and the two saddles (engine core, closed-form) ----
    s_bh = bh_entropy_evaporating(tau, _T_EVAP)          # ∝ r_h²: the falling saddle
    s_rad, island_on = page_curve(tau, _T_EVAP)
    s_hawk = 1.0 - s_bh                                   # the rising Hawking saddle
    _ = (s_rad, island_on)                                # = (min of saddles, tau > t_page)
    t_pg = page_time(_T_EVAP)
    island = 0.5 * (1.0 + math.tanh((tau - t_pg) / 0.6))  # smooth ignition at t_page

    r_h = _R_H0 * max(1.0 - tau / _T_EVAP, 0.0) ** (1.0 / 3.0)
    if r_h > 0.05:
        m_bh = mass_of_radius(r_h, _L_ADS)
        r_cap = horizon_radius(m_bh, _L_ADS) * 1.02
        # small holes are HOT (T ~ 1/r_h): the end-of-life flash — clamped for rendering
        t_hawk = min(hawking_temperature(m_bh, _L_ADS), 2.0 * hawking_page_temperature(_L_ADS))
    else:
        m_bh, r_cap = 0.0, 0.0
        t_hawk = 2.0 * hawking_page_temperature(_L_ADS)   # the final flash cooling off
    qes_r = r_cap * 1.10                                  # the QES hugs the horizon

    az = float(time) * 0.30 + float(mouse[0]) * 0.006
    dist = 22.0
    eye = wp.vec3(dist * math.sin(az), 2.4, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(75.0), float(time),
                      max_steps, max_bounce, float(m_bh), float(r_cap), float(t_hawk * 2.5),
                      float(s_hawk), float(island), float(qes_r)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ads_page_curve",
    description="the Page curve watched in real time — a black hole evaporates on the "
                "honest Stefan-Boltzmann schedule r_h ~ (1-t/T)^(1/3), heating as it "
                "shrinks, while its radiation fills the AdS box; at the closed-form Page "
                "time T(1-2^(-3/2)) the island saddle wins the path integral: a violet "
                "quantum extremal surface ignites at the horizon and the radiation "
                "purifies from thermal orange to coherent cyan — unitarity restored by "
                "saddle competition, the same minimum rule as the mutual-information and "
                "Hawking-Page transitions. --frames runs one full evaporation.",
    renderer=_render,
)
