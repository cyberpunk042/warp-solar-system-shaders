"""Inside the AdS box — a Schwarzschild-AdS black hole ray-traced to its holographic boundary.

The engine-level half of the holography set (the ``ads_cft`` disk is the map; this is the
territory). The camera floats **inside** global Anti-de Sitter space with a black hole at the
centre, and every camera ray is a photon integrated through the curved bulk:

* **Honest geodesics.** In Schwarzschild-AdS the photon orbital equation is
  ``d²u/dφ² + u = 3Mu²`` — the cosmological constant *drops out of the path shape* (Islam
  1983), so the bulk uses the proven null-geodesic pull ``a = −(3/2)h²x/r⁵`` shared with
  gargantua/kerr (``engine.blackhole``). Shadow, photon ring and lensed disk are all real.
* **The AdS box.** What Λ *does* change: the conformal boundary is timelike and sits at
  finite optical distance — light reaches it and **reflects back in**. Rays bounce off the
  boundary sphere (count set by ``--quality``), so the hole and its disk re-appear as
  boundary-mirrored images: AdS as a resonant box, the reason bulk physics is recorded on
  the boundary.
* **The CFT on the boundary.** Each boundary hit paints the conformal lattice
  (``engine.adscft.boundary_cft`` — the same `{7,3}` fold as the disk scene, stereographic =
  conformal), with a **thermal wash set by the hole's Hawking temperature**
  ``T = f'(r_h)/4π`` — the bulk black hole *is* a thermal state of the boundary theory
  (Hawking & Page). Emission is normalized at the cutoff radius — holographic
  renormalization, literally.

Honours ``--quality`` (integration steps + boundary bounces scale with the LOD tier). See
``docs/research/46-ads-cft-holography.md``. Orbit with ``--frames``; iMouse orbits.
"""

import math

import warp as wp

from .. import lod
from ..engine import post
from ..engine.adscft import ads_blackening, boundary_cft, hawking_temperature
from ..engine.blackhole import disk_emission
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

# geometry in units 2M = 1 (horizon of the Λ=0 hole at r = 1, like gargantua)
_M_BH = 0.5
_L_ADS = wp.constant(7.0)                 # AdS curvature radius
_R_BDY = wp.constant(14.0)                # conformal-boundary cutoff (the renormalization scale)
_R_IN = wp.constant(2.6)
_R_OUT = wp.constant(9.0)
_T_HAWK = hawking_temperature(_M_BH, 7.0)  # thermal state of the boundary theory

_BOUNCES = {"low": 1, "medium": 2, "high": 3, "ultra": 4}


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float, max_steps: int, max_bounce: int, t_hawk: float):
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

    for _s in range(max_steps):
        r = wp.length(pos)
        if r < 1.02:                       # captured by the horizon — the shadow
            break

        if r > _R_BDY:                     # the timelike conformal boundary: emit CFT + reflect
            n = pos / r
            col = col + boundary_cft(n, time, t_hawk) * trans
            bounce += 1
            if bounce > max_bounce:
                break
            trans = trans * 0.42
            vel = vel - n * (2.0 * wp.dot(vel, n))
            pos = n * (_R_BDY - 1.0e-3)
            cr2 = wp.cross(pos, vel)       # h is preserved by the mirror, but re-derive safely
            h2 = wp.dot(cr2, cr2)

        prev = pos
        # Schwarzschild null-geodesic pull — exact photon path shape in Schwarzschild-AdS too,
        # since Lambda drops out of d²u/dφ² + u = 3Mu².
        acc = pos * (-1.5 * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.45)
        vel = vel + acc * dt
        pos = pos + vel * dt

        if prev[1] * pos[1] < 0.0:         # equatorial accretion-disk crossing
            f = prev[1] / (prev[1] - pos[1])
            cp = prev + (pos - prev) * f
            emit = disk_emission(cp, wp.normalize(vel), time, _R_IN, _R_OUT, 7200.0, 0.60)
            if emit[0] + emit[1] + emit[2] > 0.0:
                # extra AdS redshift between emission radius and camera: sqrt(f_em/f_cam)
                rc = wp.length(cp)
                g = wp.sqrt(ads_blackening(rc, _L_ADS, 0.5) /
                            ads_blackening(wp.length(eye), _L_ADS, 0.5))
                col = col + emit * (trans * g)
                trans = trans * 0.35

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = lod.active_tier()
    max_steps = tier.raymarch_steps * 4
    max_bounce = _BOUNCES.get(tier.name, 2)

    az = float(time) * 0.42 + float(mouse[0]) * 0.006
    dist = 8.5
    eye = wp.vec3(dist * math.sin(az), 1.8, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(52.0), float(time),
                      max_steps, max_bounce, float(_T_HAWK)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ads_bulk",
    description="inside the AdS box — a Schwarzschild-AdS black hole ray-traced with real null "
                "geodesics to the timelike conformal boundary, which reflects light back in "
                "(bounces scale with --quality) so the hole and disk repeat as boundary-mirrored "
                "images; every boundary hit paints the CFT lattice with a thermal wash set by "
                "the hole's Hawking temperature — the thermal state dual to the black hole.",
    renderer=_render,
)
