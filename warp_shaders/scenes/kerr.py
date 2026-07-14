"""Kerr — a *spinning* black hole, where spacetime itself is dragged around.

Gargantua's hole was static (Schwarzschild). A real astrophysical black hole spins, and rotation
warps the geometry in two visible ways this scene ray-traces from the photon paths up:

* **Frame-dragging (the Lense–Thirring effect).** A rotating mass drags the spacetime around it
  like a whirlpool, so even light is swept along the spin. We add the gravitomagnetic force
  ``a = κ · v × B_g`` with the spin dipole field ``B_g ∝ [3(Ĵ·r̂)r̂ − Ĵ] / r³`` on top of the
  Schwarzschild pull. Photons co-rotating with the hole (the *prograde* side) are pulled in more
  easily, photons counter-rotating are flung wide — so the black shadow becomes **asymmetric**,
  flattened on one edge, the signature "D" of a Kerr hole.
* **Extreme one-sided Doppler.** Spin lets the innermost stable orbit sit much closer and move
  much faster than around a static hole, so the disk's approaching edge is beamed into a
  brilliant blue-white blade while the receding edge sinks to a dim ember — far more lopsided
  than Gargantua's.

Same physical accretion disk (Shakura–Sunyaev blackbody + relativistic beaming + gravitational
redshift, ``engine.blackhole.disk_emission``) and lensed starfield. See
``docs/research/43-relativistic-masterpieces.md``.
"""

import math

import warp as wp

from ..engine import post
from ..engine.blackhole import cosmic_background, disk_emission
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_R_IN = wp.constant(2.3)          # prograde ISCO sits closer for a spinning hole
_R_OUT = wp.constant(11.0)
_R_ESC = wp.constant(45.0)
_SPIN = wp.constant(1.05)         # dimensionless-ish drag strength (spin along +y, the disk axis)
_MAXSTEP = 620


@wp.func
def _accel(pos: wp.vec3, vel: wp.vec3, h2: float) -> wp.vec3:
    r = wp.length(pos)
    # Schwarzschild null-geodesic pull toward the centre
    grav = pos * (-1.5 * h2 / (r * r * r * r * r))
    # gravitomagnetic (frame-dragging) dipole of a mass spinning about +y
    jhat = wp.vec3(0.0, 1.0, 0.0)
    rhat = pos / r
    bg = (rhat * (3.0 * wp.dot(jhat, rhat)) - jhat) / (r * r * r)
    drag = wp.cross(vel, bg) * (_SPIN * h2)
    return grav + drag


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
    h2 = wp.dot(cr, cr)

    col = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    captured = int(0)

    for _s in range(_MAXSTEP):
        r = wp.length(pos)
        if r < 1.02:
            captured = 1
            break
        if r > _R_ESC:
            break
        prev = pos
        acc = _accel(pos, vel, h2)
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.34)
        vel = vel + acc * dt
        pos = pos + vel * dt
        if prev[1] * pos[1] < 0.0:
            f = prev[1] / (prev[1] - pos[1])
            cp = prev + (pos - prev) * f
            emit = disk_emission(cp, wp.normalize(vel), time, _R_IN, _R_OUT, 6800.0, 0.30)
            if emit[0] + emit[1] + emit[2] > 0.0:
                col = col + emit * trans
                trans = trans * 0.34

    if captured == 0:
        col = col + cosmic_background(wp.normalize(vel), 0.0) * trans

    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = float(time) * 1.05 + float(mouse[0]) * 0.006
    dist = 13.0
    eye = wp.vec3(dist * math.sin(az), 1.65, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(34.0), float(time)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="kerr",
    description="a spinning (Kerr) black hole ray-traced with frame-dragging — the Lense-Thirring "
                "gravitomagnetic force twists photon paths so the event-horizon shadow is skewed "
                "into the asymmetric 'D' of a rotating hole, while the closer, faster prograde "
                "disk edge is Doppler-beamed into a blue-white blade against a lensed starfield.",
    renderer=_render,
)
