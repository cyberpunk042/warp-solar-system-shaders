"""ER = EPR — the eternal black hole as a wormhole between two entangled universes.

The eternal Schwarzschild-AdS black hole has TWO asymptotic boundaries, and the duality
reads it as two copies of the CFT entangled in the **thermofield-double state**
``|TFD⟩ = Σ e^{−βE/2}|E⟩_L|E⟩_R`` (Maldacena 2001). The Einstein-Rosen bridge between the
exteriors IS that entanglement — Maldacena & Susskind's ER=EPR. But the bridge is
non-traversable: the two universes share a state, not a channel; you cannot send a signal
through entanglement alone.

Gao, Jafferis & Wall (2016) found the loophole: couple the two boundary theories directly
(``δH = −g O_L O_R``, a double-trace deformation). The coupling injects **negative null
energy** into the bulk, the horizon shrinks back, and the wormhole opens — briefly,
traversably. Entanglement plus classical coupling = a channel: this is exactly bulk
**quantum teleportation** through the ER bridge.

The scene runs that protocol as a cycle, from our side (the warm boundary):

* **Coupling OFF** — an honest eternal hole. Null geodesics (``a = −3Mh²x/r⁵``, exact in
  Schwarzschild-AdS) fall through the photon ring into a pure shadow: the other universe
  is *there* (the TFD state knows it) but causally out of reach. Our boundary lattice
  (``boundary_cft``) reflects and glows, orange and thermal.
* **Coupling ON** — the throat opens. Rays that would have been captured now cross the
  bridge and climb out the far side, and the shadow **fills with the other universe's
  CFT** — the cool counter-rotating lattice of ``boundary_cft_dual`` (``H_L = −H_R``: the
  two copies flow oppositely in the TFD). A blue negative-energy wash marks the open
  throat. What was the darkest region of the sky becomes a window.

The traversal itself is schematic (the ray is handed through the throat rather than
integrated through the GJW-deformed metric); everything else — the exterior geodesics,
the reflecting boundary, the two entangled CFT copies and the coupling gate — is the
honest dictionary. See ``docs/research/46-ads-cft-holography.md`` (Part V).
--frames runs the coupling cycle; iMouse orbits.
"""

import math

import warp as wp

from .. import lod
from ..engine import post
from ..engine.adscft import boundary_cft, boundary_cft_dual, hawking_temperature
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_M_BH = 0.5
_L_ADS = 7.0
_R_BDY = wp.constant(14.0)               # conformal-boundary cutoff (both sides)
_T_HAWK = hawking_temperature(_M_BH, _L_ADS)
_BOUNCES = {"low": 1, "medium": 2, "high": 3, "ultra": 4}
_OMEGA = 0.5                             # coupling cycle rate: ON/OFF period ~ 12.6 s


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float, max_steps: int, max_bounce: int, t_hawk: float,
                   coupling: float):
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
    side = float(0.0)                    # 0 = our universe (L), 1 = beyond the bridge (R)

    for _s in range(max_steps):
        r = wp.length(pos)

        if r < 1.02:
            if coupling < 0.01 or side > 0.5:
                break                    # non-traversable: the shadow — ER without the channel
            # the GJW protocol: negative null energy holds the throat open and the ray
            # is handed through the bridge into the other exterior (schematic traversal)
            side = 1.0
            col = col + wp.vec3(0.25, 0.45, 1.00) * (0.22 * trans * coupling)
            trans = trans * (0.55 + 0.35 * coupling)
            n = pos / r
            # emerge from the far mouth, still moving the same way: in through one mouth,
            # out the other (the antipodal hand-off) — the infalling ray is now a climbing one
            pos = n * (-1.06)
            cr2 = wp.cross(pos, vel)
            h2 = wp.dot(cr2, cr2)

        if r > _R_BDY:
            n = pos / r
            if side > 0.5:
                # the OTHER universe's boundary: the dual CFT copy, seen through the bridge
                col = col + boundary_cft_dual(n, time, t_hawk) * (trans * coupling)
                break
            col = col + boundary_cft(n, time, t_hawk) * trans
            bounce += 1
            if bounce > max_bounce:
                break
            trans = trans * 0.42
            vel = vel - n * (2.0 * wp.dot(vel, n))
            pos = n * (_R_BDY - 1.0e-3)
            cr2 = wp.cross(pos, vel)
            h2 = wp.dot(cr2, cr2)

        # exact Schwarzschild-AdS photon pull (Lambda drops out of the path shape)
        acc = pos * (-1.5 * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.45)
        vel = vel + acc * dt
        pos = pos + vel * dt

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = lod.active_tier()
    max_steps = tier.raymarch_steps * 4
    max_bounce = _BOUNCES.get(tier.name, 2)

    # the double-trace coupling g(t): smoothly gated ON/OFF protocol cycle
    coupling = 0.5 * (1.0 + math.tanh(3.0 * math.sin(_OMEGA * float(time))))

    az = float(time) * 0.30 + float(mouse[0]) * 0.006
    dist = 8.5
    eye = wp.vec3(dist * math.sin(az), 1.8, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(52.0), float(time),
                      max_steps, max_bounce, float(_T_HAWK), float(coupling)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ads_wormhole",
    description="ER=EPR and the Gao-Jafferis-Wall protocol — the eternal AdS black hole as "
                "a thermofield-double wormhole between two entangled CFTs: coupling OFF, an "
                "honest non-traversable shadow (entanglement is not a channel); coupling ON, "
                "negative null energy opens the throat and the shadow fills with the OTHER "
                "universe's counter-rotating cyan lattice — quantum teleportation drawn as a "
                "window in the darkest part of the sky. --frames runs the coupling cycle.",
    renderer=_render,
)
