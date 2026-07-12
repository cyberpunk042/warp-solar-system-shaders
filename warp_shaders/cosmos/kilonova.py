"""Kilonova — a neutron-star merger and its r-process ejecta.

Two neutron stars **inspiral** (chirping inward), **merge** in a flash, and throw
off neutron-rich ejecta whose radioactive r-process decay glows as a **kilonova**:
a fast **blue** polar component (lanthanide-poor) and a slower **red** equatorial
one (lanthanide-rich), plus a brief collimated **short-gamma-ray-burst jet** along
the poles. See ``docs/research/20-more-cosmos-worlds-crossstrand.md`` (GW170817).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3


@wp.func
def _point_glow(ro: wp.vec3, rd: wp.vec3, ctr: wp.vec3, w: float) -> float:
    b = wp.length(wp.cross(ctr - ro, rd))               # impact parameter (rd unit)
    return wp.exp(-(b * b) / w)


@wp.func
def _kn_at(p: wp.vec3, r_pol: float, r_eq: float, time: float) -> wp.vec4:
    """Two-component kilonova ejecta emission at `p`."""
    r = wp.length(p)
    dir = p / wp.max(r, 1.0e-4)
    costh = wp.abs(dir[1])                               # 1 at poles, 0 at equator
    turb = 0.45 + 0.7 * fbm3(dir * 5.0 + wp.vec3(0.0, time * 0.1, 0.0), 4)
    # blue polar shell (fast, lanthanide-poor)
    dp = (r - r_pol) / (0.16 * r_pol + 0.05)
    pol = wp.exp(-dp * dp) * wp.smoothstep(0.25, 0.9, costh)
    # red equatorial shell (slower, lanthanide-rich)
    de = (r - r_eq) / (0.18 * r_eq + 0.05)
    eq = wp.exp(-de * de) * wp.smoothstep(0.75, 0.2, costh)
    blue = wp.vec3(0.45, 0.68, 1.15) * pol
    red = wp.vec3(1.15, 0.35, 0.12) * eq
    col = (blue + red) * turb
    dens = (pol + eq) * turb
    return wp.vec4(col[0], col[1], col[2], dens)


@wp.kernel
def kilonova_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, phase: int,
                    orbit_r: float, spin: float, r_pol: float, r_eq: float,
                    jet: float, ns_glow: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)
    col = stars(rd)

    if phase == 0:
        # inspiral: two neutron stars orbiting, chirping inward
        a = wp.vec3(wp.cos(spin) * orbit_r, 0.0, wp.sin(spin) * orbit_r)
        g = _point_glow(ro, rd, a, 0.05) + _point_glow(ro, rd, a * (-1.0), 0.05)
        col = col + wp.vec3(0.7, 0.85, 1.2) * (g * ns_glow)
    else:
        # expanding two-component ejecta (bounded volume march)
        bound = wp.max(r_pol, r_eq) * 1.5
        b = wp.dot(ro, rd)
        c = wp.dot(ro, ro) - bound * bound
        disc = b * b - c
        if disc > 0.0:
            sq = wp.sqrt(disc)
            t0 = wp.max(-b - sq, 0.0)
            t1 = -b + sq
            seg = (t1 - t0) / 40.0
            tt = t0 + 0.5 * seg
            trans = float(1.0)
            acc = wp.vec3(0.0, 0.0, 0.0)
            for _ in range(40):
                nv = _kn_at(ro + rd * tt, r_pol, r_eq, time)
                dn = nv[3] * seg * 1.6
                if dn > 0.001:
                    acc = acc + wp.vec3(nv[0], nv[1], nv[2]) * (dn * trans)
                    trans = trans * (1.0 - wp.clamp(dn, 0.0, 1.0))
                tt += seg
            col = col + acc
        # short-gamma-ray-burst jet: a thin bright cone along +/- y
        if jet > 0.0:
            beam = wp.pow(wp.clamp(wp.abs(rd[1]), 0.0, 1.0), 40.0)   # thin polar cone
            col = col + wp.vec3(0.7, 0.85, 1.2) * (beam * jet)

    img[i, j] = col


def render_kilonova(width, height, time, mouse, device, period=12.0):
    """Render one frame — inspiral, merge flash, then the two-colour ejecta + jet."""
    prog = time / period
    t_merge = 0.42
    az = 0.6 + float(mouse[0]) * 0.01
    elev = 0.35 + float(mouse[1]) * 0.01                 # tilt to show poles vs equator
    dist = 15.0
    eye = (math.sin(az) * dist * math.cos(elev), math.sin(elev) * dist,
           math.cos(az) * dist * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=48.0, aspect=width / height)

    if prog < t_merge:
        phase = 0
        u = prog / t_merge
        orbit_r = 3.2 * (1.0 - u) ** 0.5 + 0.25          # shrinks toward merge
        spin = 18.0 * (u ** 1.6)                         # chirp: spins ever faster
        r_pol = 0.0
        r_eq = 0.0
        jet = 0.0
        ns_glow = 1.2 + 2.0 * u                          # brightens as they near
    else:
        phase = 1
        u = (prog - t_merge) / (1.0 - t_merge)
        orbit_r = 0.0
        spin = 0.0
        r_pol = 0.6 + 10.0 * u                           # polar ejecta is faster
        r_eq = 0.5 + 6.5 * u
        jet = max(0.0, 2.5 * math.exp(-8.0 * u))         # brief GRB jet
        ns_glow = 0.0

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(kilonova_kernel, dim=(height, width),
              inputs=[img, cam, int(phase), float(orbit_r), float(spin), float(r_pol),
                      float(r_eq), float(jet), float(ns_glow), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    # merge flash
    if abs(prog - t_merge) < 0.06:
        f = 1.0 - abs(prog - t_merge) / 0.06
        hdr = hdr + np.array([0.85, 0.92, 1.1], np.float32) * (f * f * 3.5)

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)
