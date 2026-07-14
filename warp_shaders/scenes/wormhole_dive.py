"""Wormhole dive — falling through an Ellis throat into another universe, by real geodesics.

Where the older ``wormhole`` scene paints an analytic portal, this one **integrates the metric**.
The Ellis–Morris-Thorne wormhole ``ds² = -dt² + dℓ² + (b₀²+ℓ²)(dθ² + sin²θ dφ²)`` is a tunnel
whose areal radius ``r = √(b₀²+ℓ²)`` never shrinks below the throat ``b₀`` — the signed
coordinate ``ℓ`` runs from ``-∞`` (our universe) through ``0`` (the throat) to ``+∞`` (another).
Every camera ray is a photon carrying a conserved angular momentum ``L``; we integrate its null
geodesic ``ℓ'' = L²ℓ/r⁴`` in the ray's plane. Rays with a small impact parameter thread the
throat and come out the *other side* — so a whole second universe is fish-eyed into a disc in the
middle of the view — while grazing rays turn back and lens our own sky into a bright **Einstein
ring** around the exotic-matter mouth.

Over ``--frames`` the camera **dives**: ``ℓ`` slides from deep in our universe, through the
throat, and out the far side — the second universe swelling from a coin to the whole sky as you
cross. A true fly-through, not a texture trick. See ``docs/research/43-relativistic-masterpieces.md``.
"""

import math

import warp as wp

from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.pathtrace import camera_basis, tanfov
from ..engine.sky import milky_way, starfield
from ..scene import Scene

_B0 = wp.constant(1.35)           # throat radius
_ELL_ESC = wp.constant(34.0)      # |ℓ| at which the ray has escaped to a flat asymptotic region
_MAXSTEP = 1200


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   ell0: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanf * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanf
    rd = wp.normalize(fwd + right * u + up * v)

    e0 = wp.normalize(eye)                              # outward radial unit in the current universe
    cpsi = -wp.dot(rd, e0)                             # cos angle from the inward (toward-throat) axis
    tvec = rd - e0 * wp.dot(rd, e0)
    tl = wp.length(tvec)
    e1 = tvec / wp.max(tl, 1e-6)                        # tangential unit (direction of +φ)
    spsi = tl                                          # sin of that angle (≥0)

    r0 = wp.sqrt(_B0 * _B0 + ell0 * ell0)
    ell_sign = float(1.0)
    if ell0 < 0.0:
        ell_sign = -1.0
    ell = ell0
    pl = -ell_sign * cpsi                              # radial momentum (toward throat)
    lang = r0 * spsi                                   # conserved angular momentum L
    phi = float(0.0)
    l2 = lang * lang
    rmin = r0

    for _s in range(_MAXSTEP):
        r2 = _B0 * _B0 + ell * ell
        r = wp.sqrt(r2)
        if r < rmin:
            rmin = r
        if wp.abs(ell) > _ELL_ESC:
            break
        h = wp.clamp(0.14 * r, 0.02, 0.5)
        pl = pl + (l2 * ell / (r2 * r2)) * h
        ell = ell + pl * h
        phi = phi + (lang / r2) * h

    r2 = _B0 * _B0 + ell * ell
    r = wp.sqrt(r2)
    drdl = (ell / r) * pl
    dphi = lang / r2
    d = (e0 * wp.cos(phi) + e1 * wp.sin(phi)) * drdl \
        + (e0 * (-wp.sin(phi)) + e1 * wp.cos(phi)) * (r * dphi)
    d = wp.normalize(d)

    if ell > 0.0:                                      # emerged in the OTHER universe (warm/amber)
        col = wp.cw_mul(starfield(d), wp.vec3(1.35, 0.95, 0.6))
        col = col + milky_way(d, wp.vec3(1.0, 0.55, 0.2), 0.55)
    else:                                              # our universe (cool/blue)
        col = starfield(d)
        col = col + milky_way(d, wp.vec3(0.35, 0.6, 1.0), 0.45)
        # exotic-matter mouth: a thin bright Einstein ring, only on rays that grazed the throat
        # and turned back (through-rays keep the far universe clean)
        g = (rmin - _B0) / 0.22
        col = col + kelvin_to_rgb(8000.0) * (wp.exp(-g * g) * 0.4)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    ell0 = -8.5 + float(time) * 1.7 + float(mouse[1]) * 0.01     # dive: our universe → throat → far side
    r0 = math.sqrt(1.35 * 1.35 + ell0 * ell0)
    az = 0.0 + float(mouse[0]) * 0.006
    rhat = wp.vec3(math.sin(az), 0.12, -math.cos(az))
    eye = wp.vec3(r0 * rhat[0], r0 * rhat[1], r0 * rhat[2])
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))    # look toward the throat

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(52.0), float(ell0)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="wormhole_dive",
    description="a geodesic fly-through of an Ellis (Morris-Thorne) wormhole — camera rays are "
                "photons integrated through the throat metric, so a second amber universe is "
                "fish-eyed through the mouth while our own blue sky lenses into an Einstein ring "
                "around the exotic-matter rim; over frames the camera dives across the throat.",
    renderer=_render,
)
