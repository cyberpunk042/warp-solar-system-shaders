"""Tidal disruption event — a star spaghettified and devoured by a black hole.

Reuses the black-hole GR photon integrator (bent light + lensed horizon). Instead
of a steady accretion disk, the equatorial plane carries a **debris stream**: a
single trailing log-spiral arm of gas torn from the star, hot blue-white where it
plunges toward the hole and cooling to the star's orange at the trailing end. Over
`time` the stream **grows** (the star stretches) and a central **flare** brightens
as the debris accretes. See ``docs/research/19-extraordinary-cosmos.md``.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from .blackhole import bh_escape_dir

_STEPS = 220


@wp.func
def _stream_emission(cp: wp.vec3, rs: float, prog: float, time: float) -> wp.vec3:
    """Debris-stream emission where the bent ray crosses the equatorial plane."""
    r = wp.sqrt(cp[0] * cp[0] + cp[2] * cp[2])
    r_in = 1.8 * rs
    r_out = 14.0 * rs
    ang = wp.atan2(cp[2], cp[0])
    # a trailing log-spiral arm: the stream winds inward, rotating in time
    arm_phase = 2.3 * wp.log(r / rs + 0.5) - time * 0.5
    d = wp.sin(ang - arm_phase)
    arm = wp.exp(-(d * d) / 0.14)                        # broad gaussian around the arm
    # the stream reaches from the hole out to a tip that extends as the star is drawn in
    extent = r_in + (r_out - r_in) * wp.clamp(0.25 + 0.75 * prog, 0.0, 1.0)
    band = wp.smoothstep(extent, extent * 0.8, r) * wp.smoothstep(r_in * 0.6, r_in, r)
    turb = 0.55 + 0.7 * fbm3(cp * (0.5 / rs) + wp.vec3(0.0, time * 0.3, 0.0), 3)
    dens = arm * band * turb
    if dens < 0.001:
        return wp.vec3(0.0, 0.0, 0.0)
    tnorm = wp.clamp((r - r_in) / (r_out - r_in), 0.0, 1.0)
    hot = wp.vec3(0.85, 0.93, 1.15)                      # plunging gas: hot blue-white
    warm = wp.vec3(1.05, 0.5, 0.18)                      # trailing end: the star's orange
    col = hot * (1.0 - tnorm) + warm * tnorm
    bright = dens * (1.5 - 1.1 * tnorm)                  # far brighter near the hole
    return col * bright


@wp.func
def tde_pixel(ro: wp.vec3, rd: wp.vec3, rs: float, prog: float,
              time: float) -> wp.vec4:
    p = ro
    v = rd
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
        # central accretion flare — a compact hot glow that swells as debris feeds
        # the hole (tight gaussian so it does not haze the whole frame)
        fr = wp.length(p)
        flare = wp.exp(-(fr * fr) / (2.0 * 1.4 * rs * 1.4 * rs)) * (0.1 + 0.7 * prog)
        acc = acc + wp.vec3(0.8, 0.9, 1.2) * (flare * trans * dt * 0.5)
        # equatorial debris-stream crossing
        if (p[1] * pn[1]) < 0.0:
            k = p[1] / (p[1] - pn[1])
            cp = p + (pn - p) * k
            em = _stream_emission(cp, rs, prog, time)
            al = wp.clamp(0.35 + 0.4 * wp.length(em), 0.1, 1.0)
            acc = acc + em * (trans * al)
            trans = trans * (1.0 - 0.5 * al)
        p = pn
        if wp.length(p) > 60.0 * rs:
            break
    w = trans
    if captured == 1:
        w = -1.0
    return wp.vec4(acc[0], acc[1], acc[2], w)


@wp.kernel
def tde_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, rs: float, prog: float,
               time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)
    dv = tde_pixel(ro, rd, rs, prog, time)
    emit = wp.vec3(dv[0], dv[1], dv[2])
    w = dv[3]
    if w < 0.0:
        col = emit
    else:
        bent = bh_escape_dir(ro, rd, rs)
        col = stars(bent) * w + emit
    img[i, j] = col


def render_tde(width, height, time, mouse, device, rs=1.0, period=12.0):
    """Render one frame of a tidal disruption event — a star drawn into a hot
    debris stream + central flare, disruption progressing over `period` seconds."""
    prog = min(time / period, 1.0)
    az = 0.7 + time * 0.04 + float(mouse[0]) * 0.01
    elev = 0.5 + float(mouse[1]) * 0.01                  # look down onto the stream
    dist = 26.0 * rs
    eye = (math.sin(az) * dist * math.cos(elev), math.sin(elev) * dist,
           math.cos(az) * dist * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=42.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(tde_kernel, dim=(height, width),
              inputs=[img, cam, float(rs), float(prog), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.5, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=0.9)
