"""Gargantua — a black hole ray-traced by integrating real null geodesics.

Not an analytic lensing trick: every camera ray is a **photon**, and the engine integrates its
path through the curved spacetime of a Schwarzschild black hole. In geometric units (``r_s = 1``)
a light ray obeys ``d²x/dλ² = -3/2 · h² · x / r⁵`` (``h = |x × v|`` its conserved angular
momentum), which reproduces general relativity's light-bending exactly. So the ray curves: it
can fall past the **event horizon** (→ the black shadow), graze the **photon sphere** and loop
(→ the bright photon ring), or bend *over the top* of the hole and strike the far side of the
**accretion disk** — the reason the disk appears to arc above and below the shadow, the image
*Interstellar* made famous.

The disk itself is physical: a Shakura–Sunyaev temperature gradient (``T ∝ r^-3/4``) coloured by
the blackbody law, **relativistically Doppler-beamed** (the side orbiting toward you brightens
and blueshifts, the receding side dims and reddens) and **gravitationally redshifted** near the
hole. The background is a real starfield, **lensed** because the escaping rays were bent. See
``docs/research/42-gravitational-lensing.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.color import kelvin_to_rgb
from ..engine.pathtrace import camera_basis, tanfov
from ..engine.sky import starfield
from ..scene import Scene

_R_IN = wp.constant(3.2)          # inner disk edge (~ISCO, r_s=1)
_R_OUT = wp.constant(11.0)        # outer disk edge
_R_ESC = wp.constant(45.0)        # ray escaped to infinity
_MAXSTEP = 520


@wp.func
def _disk(cp: wp.vec3, pdir: wp.vec3, time: float) -> wp.vec3:
    rc = wp.sqrt(cp[0] * cp[0] + cp[2] * cp[2])
    if rc < _R_IN or rc > _R_OUT:
        return wp.vec3(0.0, 0.0, 0.0)

    # Shakura-Sunyaev temperature gradient (warm-biased for the Gargantua gold look)
    temp = 4300.0 * wp.pow(_R_IN / rc, 0.6)
    # Keplerian orbital speed (GM = r_s/2 = 0.5) and prograde tangent direction
    beta = wp.min(wp.sqrt(0.5 / rc), 0.62)
    cph = cp / wp.length(cp)
    tang = wp.normalize(wp.cross(wp.vec3(0.0, 1.0, 0.0), cph))      # orbital velocity direction
    # Doppler: material moving toward the incoming photon (−pdir) beams + blueshifts
    approach = wp.dot(tang, -pdir)
    dopp = 1.0 / wp.max(1.0 - beta * approach, 0.2)
    # gravitational redshift factor (r_s = 1)
    grav = wp.sqrt(wp.max(1.0 - 1.0 / rc, 0.02))

    col = kelvin_to_rgb(wp.clamp(temp * dopp * grav, 1200.0, 40000.0))

    # turbulent Keplerian bands (inner rotates faster) + inner-edge brightening
    ang = wp.atan2(cp[2], cp[0]) + time * (2.4 / wp.pow(rc, 1.5))
    tex = 0.62 + 0.30 * wp.sin(ang * 5.0 + rc * 2.6) + 0.16 * wp.sin(ang * 13.0 - rc * 1.7)
    fall = wp.pow(_R_IN / rc, 1.3)
    edge = 1.0 + 1.8 * wp.exp(-(rc - _R_IN) * 1.3)                  # bright inner rim
    beam = wp.min(dopp * dopp * dopp, 6.0)                          # cap relativistic beaming
    bright = fall * edge * wp.max(tex, 0.0) * beam * grav
    return col * (bright * 0.3)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanf * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanf
    rd = wp.normalize(fwd + right * u + up * v)

    pos = eye
    vel = rd
    cr = wp.cross(pos, vel)
    h2 = wp.dot(cr, cr)                          # conserved angular momentum²

    col = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)                           # remaining transmittance (disk is semi-transparent)
    captured = int(0)

    for _s in range(_MAXSTEP):
        r = wp.length(pos)
        if r < 1.02:
            captured = 1
            break
        if r > _R_ESC:
            break
        prev = pos
        acc = pos * (-1.5 * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.018, 0.35)
        vel = vel + acc * dt
        pos = pos + vel * dt
        # equatorial-plane (y=0) crossing → the accretion disk
        if prev[1] * pos[1] < 0.0:
            f = prev[1] / (prev[1] - pos[1])
            cp = prev + (pos - prev) * f
            emit = _disk(cp, wp.normalize(vel), time)
            if emit[0] + emit[1] + emit[2] > 0.0:
                col = col + emit * trans
                trans = trans * 0.35             # each disk pass dims what's behind it

    if captured == 0:
        col = col + wp.cw_mul(starfield(wp.normalize(vel)), wp.vec3(1.0, 1.0, 1.0)) * trans

    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = float(time) * 1.15 + float(mouse[0]) * 0.006     # camera orbits over --frames
    dist = 13.0
    eye = wp.vec3(dist * math.sin(az), 1.35, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(34.0), float(time)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="gargantua",
    description="a Schwarzschild black hole ray-traced by integrating real null geodesics — the "
                "accretion disk lensed up and over the shadow (Interstellar-style), a bright "
                "photon ring, a Shakura-Sunyaev blackbody temperature gradient with relativistic "
                "Doppler beaming and gravitational redshift, over a starfield lensed by the "
                "bending of the escaping rays.",
    renderer=_render,
)
