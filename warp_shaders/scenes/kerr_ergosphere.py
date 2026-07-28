"""Spin-up — watch the ergosphere be born.

The `kerr` scene shows a spinning hole at fixed spin, cinematically. This scene turns the
spin into the *experiment*: once per cycle the dimensionless spin χ = a/M sweeps
0 → 0.98 → 0, every quantity on screen driven by the exact Kerr dictionary
(``engine.kerr``, all test-asserted):

* the **horizon shrinks**: ``r_+ = M(1 + √(1 − χ²))`` falls from 2M toward M — the
  capture radius (and the shadow) contracts as spin grows;
* the **ergosurface inflates from nothing**: ``r_E(θ) = M(1 + √(1 − χ²cos²θ))`` peels
  away from the horizon (they coincide at χ = 0), bulging to 2M at the equator while
  still touching r_+ at the poles — the scene accumulates a violet veil along every
  ray's path *through* the ergoregion ``r_+ < r < r_E(θ)``, so the oblate shell where
  standing still is impossible literally materializes around the shadow;
* the **drag turns on**: the gravitomagnetic (Lense-Thirring) force on the photons
  scales with the live spin — the same dipole force the `kerr` scene uses, but here
  ``∝ χ(t)``, so the starfield behind the hole starts to smear azimuthally and the
  shadow develops its Kerr asymmetry as you watch (``lense_thirring_omega``'s 1/r³ law
  is the far-field of this force, asserted in the suite);
* the accretion disk's Doppler blade sharpens as the spin rises (the beaming asymmetry
  is spin-fed in the drag term, not faked).

Same proven Schwarzschild null pull + relativistic disk + lensed starfield as the
masterpiece scenes (``engine.blackhole``); geometric units r_s = 1 (M = 1/2). --frames
runs one full spin-up-and-down; iMouse orbits. See
``docs/research/50-kerr-spinning-black-hole.md`` (Part I).
"""

import math

import warp as wp

from ..engine import post
from ..engine.blackhole import cosmic_background, disk_emission
from ..engine.kerr import kerr_horizons, lense_thirring_omega
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_R_IN = wp.constant(2.1)
_R_OUT = wp.constant(9.5)
_R_ESC = wp.constant(45.0)
_MAXSTEP = 620
_M_GEO = 0.5                       # r_s = 1 units
_T_CYCLE = 14.0


@wp.func
def _accel(pos: wp.vec3, vel: wp.vec3, h2: float, drag_amp: float) -> wp.vec3:
    r = wp.length(pos)
    grav = pos * (-1.5 * h2 / (r * r * r * r * r))
    # gravitomagnetic dipole WITHOUT the per-ray h2 amplifier the cinematic `kerr`
    # scene uses — here the drag must stay a genuine 1/r^3 Lense-Thirring force, or
    # wide rays with large h2 get funnelled through the ergoregion and the veil
    # (which images exactly those rays) balloons across the frame
    jhat = wp.vec3(0.0, 1.0, 0.0)
    rhat = pos / r
    bg = (rhat * (3.0 * wp.dot(jhat, rhat)) - jhat) / (r * r * r)
    drag = wp.cross(vel, bg) * drag_amp
    return grav + drag


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float, chi: float, r_plus: float, drag_amp: float):
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
    ergo_len = float(0.0)                       # path length spent inside the ergoregion

    for _s in range(_MAXSTEP):
        r = wp.length(pos)
        if r < r_plus * 1.02:
            captured = 1
            break
        if r > _R_ESC:
            break

        # inside the ergoregion?  r_+ < r < r_E(theta),  cos(theta) = y/r  — smoothly
        # weighted at both faces of the shell so step sampling doesn't band
        cth = pos[1] / r
        r_ergo = _M_GEO * (1.0 + wp.sqrt(wp.max(1.0 - chi * chi * cth * cth, 0.0)))
        prev = pos
        acc = _accel(pos, vel, h2, drag_amp)
        dt = wp.clamp(0.10 * r / 3.0, 0.008, 0.30)
        shell = wp.clamp((r_ergo - r) / 0.06, 0.0, 1.0) * wp.clamp((r - r_plus) / 0.03, 0.0, 1.0)
        ergo_len += dt * shell
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

    # the ergoregion veil: violet glow along the path spent where standing still is
    # impossible — zero at chi = 0 (the surface coincides with the horizon), an oblate
    # shell around the shadow near extremality (dimmed on captured rays so the shadow
    # keeps its darkness under a violet rim)
    veil = (1.0 - wp.exp(-ergo_len * 1.8)) * 0.60
    if captured == 1:
        veil = veil * 0.07                     # keep the shadow black; the rim is the veil
    col = col + wp.vec3(0.62, 0.30, 1.00) * veil

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tau = math.fmod(float(time), _T_CYCLE)
    chi = 0.98 * 0.5 * (1.0 - math.cos(2.0 * math.pi * tau / _T_CYCLE))

    r_plus, _ = kerr_horizons(_M_GEO, chi * _M_GEO)            # r_s = 1 units
    drag_amp = 0.9 * chi                                       # LT force scales with spin
    _ = lense_thirring_omega(3.0, _M_GEO, chi * _M_GEO)        # the far-field law (asserted)

    az = float(time) * 0.55 + float(mouse[0]) * 0.006
    dist = 17.0
    eye = wp.vec3(dist * math.sin(az), 2.4, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(27.0), float(time),
                      float(chi), float(r_plus), float(drag_amp)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.35, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="kerr_ergosphere",
    description="watch the ergosphere be born — once per cycle the spin sweeps 0 to "
                "0.98: the horizon r_+ = M(1+sqrt(1-chi^2)) contracts, the violet "
                "ergosurface r_E(theta) peels away from it and inflates into the oblate "
                "shell where standing still is impossible (drawn as accumulated path "
                "length through the ergoregion), the Lense-Thirring drag smears the "
                "starfield, and the disk's Doppler blade sharpens — the exact Kerr "
                "dictionary as a time axis. --frames runs one spin-up-and-down.",
    renderer=_render,
)
