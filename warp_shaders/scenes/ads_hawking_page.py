"""The Hawking-Page transition — thermal AdS and the large black hole trading places.

The thermodynamic heart of AdS/CFT, ray-traced. Heat the boundary theory through
``T_HP = 1/(πL)`` and the *entire bulk geometry changes phase* (Hawking & Page 1983):

* **Below T_HP — thermal AdS.** No horizon anywhere. The box is filled with a dim thermal
  graviton gas and the reflecting conformal boundary: rays bounce until absorbed, the CFT
  lattice glowing gently at the boundary temperature. On the boundary this is the
  **confined** phase (free energy ~ O(1)).
* **Above T_HP — the large black hole.** The dominant saddle of the canonical ensemble is
  suddenly a hole whose horizon is *already AdS-sized* (``r_h = L`` exactly at the
  transition — first-order, no gentle growth), in equilibrium with its own radiation
  (the Hartle-Hawking heat bath — no accretion disk: the box IS the bath), its Hawking
  temperature locked to the boundary temperature. This is the **deconfined**
  phase (free energy ~ O(N²)) — Witten's interpretation of the same transition in the
  gauge theory.

The scene sweeps the boundary temperature sinusoidally through T_HP, so the hole
**nucleates and evaporates once per cycle** — not by dynamical collapse but by *ensemble
dominance*: which saddle wins the partition function. All the dictionary functions are the
shared engine core (``engine.adscft``): ``hawking_page_temperature``, ``large_hole_radius``
(the stable branch inversion of ``T(r_h)``), ``mass_of_radius``, ``horizon_radius``. Null
geodesics use ``a = −3Mh²x/r⁵`` (exact in Schwarzschild-AdS — Λ drops out of the path
shape), and the boundary reflects (bounces scale with ``--quality``).

See ``docs/research/46-ads-cft-holography.md`` (Part III). --frames runs the phase cycle;
iMouse orbits.
"""

import math

import warp as wp

from .. import lod
from ..engine import post
from ..engine.adscft import (
    boundary_cft,
    hawking_page_temperature,
    hawking_temperature,
    horizon_radius,
    large_hole_radius,
    mass_of_radius,
)
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_L_ADS = 2.2                              # AdS curvature radius (also r_h at the transition)
_R_BDY = wp.constant(24.0)                # conformal-boundary cutoff (renormalization scale)
_OMEGA = 0.8                              # temperature sweep rate: one full phase cycle ~ 7.9 s
_BOUNCES = {"low": 1, "medium": 2, "high": 3, "ultra": 4}


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float, max_steps: int, max_bounce: int,
                   m_bh: float, r_cap: float, t_bdy: float, hole: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanf * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanf
    rd = wp.normalize(fwd + right * u + up * v)

    pos = eye
    vel = rd
    cr = wp.cross(pos, vel)
    h2 = wp.dot(cr, cr)

    col = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    bounce = int(0)
    gas = float(0.0)                       # thermal-graviton-gas path glow (thermal AdS phase)

    for _s in range(max_steps):
        r = wp.length(pos)
        if hole > 0.0 and r < r_cap:       # captured by the horizon — the shadow
            break

        if r > _R_BDY:                     # the timelike conformal boundary: emit CFT + reflect
            n = pos / r
            col = col + boundary_cft(n, time, t_bdy * 3.0) * (trans * 0.38)
            bounce += 1
            if bounce > max_bounce:
                break
            trans = trans * 0.42
            vel = vel - n * (2.0 * wp.dot(vel, n))
            pos = n * (_R_BDY - 1.0e-3)
            cr2 = wp.cross(pos, vel)
            h2 = wp.dot(cr2, cr2)

        prev = pos
        # exact Schwarzschild-AdS photon pull, a = -3 M h² x / r⁵ (Lambda drops out)
        acc = pos * (-3.0 * m_bh * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.5)
        vel = vel + acc * dt
        pos = pos + vel * dt
        # thermal radiation fills the box (denser near the horizon when the hole exists —
        # the Hartle-Hawking heat bath; a centred graviton gas in the thermal-AdS phase)
        gas += dt * wp.exp(-r * 0.22) * (1.0 - hole)
        if hole > 0.0:
            gas += dt * wp.exp(-(r - r_cap) * 0.8) * hole * 0.25
        _ = prev

    col = col + wp.vec3(1.00, 0.55, 0.25) * gas * t_bdy * 0.30
    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = lod.active_tier()
    max_steps = tier.raymarch_steps * 4
    max_bounce = _BOUNCES.get(tier.name, 2)

    # the boundary temperature sweeps through the Hawking-Page point
    t_hp = hawking_page_temperature(_L_ADS)
    t_bdy = t_hp * (1.0 + 0.05 * math.sin(_OMEGA * float(time)))

    # ensemble dominance: above T_HP the LARGE hole is the saddle; below, thermal AdS.
    # (`hole` is a short nucleation cross-fade — visualization smoothing of a first-order jump.)
    hole = min(max((t_bdy / t_hp - 1.0) / 0.02, 0.0), 1.0)
    if hole > 0.0:
        r_h = large_hole_radius(t_bdy, _L_ADS)
        m_bh = hole * mass_of_radius(r_h, _L_ADS)
        # at hole = 1, hawking_temperature(m_bh, L) equals t_bdy — the dial is locked
        r_cap = horizon_radius(m_bh, _L_ADS) * 1.02 if m_bh > 0.0 else 0.0
    else:
        m_bh, r_cap = 0.0, 0.0

    az = float(time) * 0.30 + float(mouse[0]) * 0.006
    dist = 22.0
    eye = wp.vec3(dist * math.sin(az), 2.4, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(75.0), float(time),
                      max_steps, max_bounce, float(m_bh), float(r_cap), float(t_bdy),
                      float(hole)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ads_hawking_page",
    description="the Hawking-Page transition ray-traced — the boundary temperature sweeps "
                "through T_HP = 1/(pi L) and the bulk changes phase by ensemble dominance: "
                "below, thermal AdS (a glowing reflecting box of graviton gas, the confined "
                "boundary phase); above, a large black hole nucleates at AdS size r_h = L, "
                "Hawking temperature locked to the boundary dial (the "
                "deconfined phase, no disk: the hole is in equilibrium with its heat bath). --frames runs the phase cycle.",
    renderer=_render,
)
