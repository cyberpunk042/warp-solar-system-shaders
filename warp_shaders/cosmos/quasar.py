"""Quasar — a supermassive black hole with relativistic bipolar jets.

Reuses the black-hole GR photon integrator: light is bent by the same
``d²x/dλ² = -3/2 h² x/|x|⁵`` deflection, the Doppler-beamed accretion disk is
crossed as in :mod:`.blackhole`, and — new here — twin **relativistic jets** are
accumulated along the bent ray: collimated cones along the spin axis radiating
**synchrotron** blue-white light, punctuated by **shock knots** drifting outward,
with the approaching jet **Doppler-beamed** brighter than its receding twin (an
active galactic nucleus / quasar core). See
``docs/research/19-extraordinary-cosmos.md``.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from .blackhole import _disk_emission, bh_escape_dir

_STEPS = 220


@wp.func
def _jet_emission(p: wp.vec3, v: wp.vec3, rs: float, time: float) -> wp.vec3:
    """Synchrotron emission of the twin jets at photon position `p` (moving `v`)."""
    rho = wp.sqrt(p[0] * p[0] + p[2] * p[2])             # distance from spin axis
    y = wp.abs(p[1])
    cone_r = 0.10 * y + 0.45 * rs                        # a narrow cone opening upward
    core = wp.exp(-(rho * rho) / (0.5 * cone_r * cone_r + 1.0e-4))
    # short enough to sit in frame; starts just above the disk
    span = wp.smoothstep(1.6 * rs, 3.0 * rs, y) * wp.smoothstep(17.0 * rs, 10.0 * rs, y)
    dens = core * span
    if dens < 0.001:
        return wp.vec3(0.0, 0.0, 0.0)
    # shock knots: bright blobs drifting outward along the jet
    knot = 0.45 + 0.55 * wp.sin(y * (1.1 / rs) - time * 3.0)
    turb = 0.65 + 0.6 * fbm3(p * (0.9 / rs) + wp.vec3(0.0, -time * 0.6, 0.0), 3)
    col = wp.vec3(0.5, 0.74, 1.3)                        # synchrotron blue-white
    bright = dens * (0.5 + 1.7 * knot) * turb
    # Doppler beaming: the jet flows outward along +/-y; the one aimed at the
    # camera (v points into it) is brighter
    ydir = p[1] / (y + 1.0e-4)                           # +1 upper jet, -1 lower
    jdir = wp.vec3(0.0, ydir, 0.0)
    beam = wp.clamp(1.0 + 1.3 * wp.dot(jdir, wp.normalize(-v)), 0.3, 2.6)
    return col * (bright * beam * 4.0)


@wp.func
def quasar_pixel(ro: wp.vec3, rd: wp.vec3, rs: float, time: float,
                 spin: float) -> wp.vec4:
    """March one photon through the lensed disk + jets. Returns (rgb, w): w<0 =
    captured (paint the emission only), w>=0 = transmittance for the background."""
    p = ro
    v = rd
    r_in = 2.3 * rs
    r_out = 6.5 * rs
    acc = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    captured = int(0)
    for _ in range(_STEPS):
        h = wp.cross(p, v)
        h2 = wp.dot(h, h)
        r2 = wp.dot(p, p)
        a = p * (-1.5 * h2 / wp.pow(r2, 2.5))
        dt = 0.10 + 0.04 * wp.sqrt(r2)
        v = v + a * dt
        pn = p + v * dt
        rn = wp.length(pn)
        if rn < rs:
            captured = 1
            break
        # jets — optically-thin emission accumulated each step
        jem = _jet_emission(p, v, rs, time)
        acc = acc + jem * (trans * dt * 0.6)
        # accretion-disk crossing (equatorial plane)
        if (p[1] * pn[1]) < 0.0:
            k = p[1] / (p[1] - pn[1])
            cp = p + (pn - p) * k
            rr = wp.sqrt(cp[0] * cp[0] + cp[2] * cp[2])
            if rr > r_in and rr < r_out:
                em = _disk_emission(cp, v, rs, r_in, r_out, time, spin)
                fade = wp.smoothstep(r_out, r_out * 0.6, rr)
                al = wp.clamp(0.5 + 0.5 * fade, 0.15, 1.0)
                acc = acc + em * (trans * al)
                trans = trans * (1.0 - al)
        p = pn
        if wp.length(p) > 60.0 * rs:
            break
    w = trans
    if captured == 1:
        w = -1.0
    return wp.vec4(acc[0], acc[1], acc[2], w)


@wp.kernel
def quasar_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, center: wp.vec3,
                  rs: float, spin: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye - center
    rd = camera_ray_dir(cam, u, vv)

    dv = quasar_pixel(ro, rd, rs, time, spin)
    emit = wp.vec3(dv[0], dv[1], dv[2])
    w = dv[3]
    if w < 0.0:
        col = emit                                       # captured: horizon + emission
    else:
        bent = bh_escape_dir(ro, rd, rs)
        col = stars(bent) * w + emit                     # lensed starfield + disk + jets
    img[i, j] = col


def render_quasar(width, height, time, mouse, device, rs=1.0):
    """Render one quasar — a lensed black hole with a Doppler disk + twin jets."""
    az = 0.6 + time * 0.05 + float(mouse[0]) * 0.01
    elev = 0.28 + float(mouse[1]) * 0.01                 # tilt so the jets read vertical
    dist = 30.0 * rs                                     # pull back so both jets fit
    eye = (math.sin(az) * dist * math.cos(elev), math.sin(elev) * dist,
           math.cos(az) * dist * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=44.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(quasar_kernel, dim=(height, width),
              inputs=[img, cam, wp.vec3(0.0, 0.0, 0.0), float(rs), 1.0,
                      float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.55, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)
