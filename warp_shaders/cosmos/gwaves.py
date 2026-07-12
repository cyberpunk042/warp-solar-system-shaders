"""Gravitational waves — a chirping binary inspiral rippling spacetime.

Two compact bodies orbit, losing energy to **gravitational radiation** and
spiralling together while the wave **chirps** (frequency + amplitude rise). The
transverse quadrupole strain is visualised as concentric **ripples** with the
m = 2 (cos 2φ) pattern that warp the background starfield — a screen-space
approximation viewed nearly face-on. Merges in a flash. See
``docs/research/20-more-cosmos-worlds-crossstrand.md`` (GW150914).
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera


@wp.kernel
def gwaves_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, freq: float,
                  omega: float, amp: float, orbit_r: float, spin: float,
                  ns_glow: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0

    r = wp.sqrt(u * u + vv * vv)
    phi = wp.atan2(vv, u)
    # quadrupole spiral ripple, decaying outward, propagating in time
    ph = freq * r - omega * time + 2.0 * phi
    env = amp / (0.25 + r)
    disp = env * wp.sin(ph) * wp.smoothstep(0.03, 0.18, r)   # no warp at the very centre
    # displace the sampling direction radially (the transverse strain stretches sky)
    inv = 1.0 / wp.max(r, 1.0e-4)
    u2 = u + disp * u * inv
    v2 = vv + disp * vv * inv
    rd = camera_ray_dir(cam, u2, v2)
    col = stars(rd)
    # faint luminous crests so the wavefronts themselves read
    col = col + wp.vec3(0.22, 0.30, 0.5) * (wp.max(wp.sin(ph), 0.0) * env * 0.5)

    # the inspiralling binary at the centre (two bright points)
    ro = cam.eye
    rdc = camera_ray_dir(cam, u, vv)
    a = wp.vec3(wp.cos(spin) * orbit_r, 0.0, wp.sin(spin) * orbit_r)
    b1 = wp.length(wp.cross(a - ro, rdc))
    b2 = wp.length(wp.cross(-a - ro, rdc))
    g = wp.exp(-(b1 * b1) / 0.04) + wp.exp(-(b2 * b2) / 0.04)
    col = col + wp.vec3(0.8, 0.88, 1.2) * (g * ns_glow)

    img[i, j] = col


def render_gwaves(width, height, time, mouse, device, period=12.0):
    """Render one frame of a chirping binary inspiral + its spacetime ripples."""
    prog = min(time / period, 1.0)
    chirp = 1.0 + 3.0 * prog ** 2                        # frequency rises toward merger
    freq = 9.0 * chirp
    omega = 2.5 * chirp
    amp = 0.05 * (0.4 + prog)                            # strain grows
    orbit_r = 0.55 * (1.0 - prog) ** 0.5 + 0.04          # orbit shrinks
    spin = 8.0 * (prog ** 1.5) * period
    ns_glow = 0.8 + 1.5 * prog

    az = 0.4 + float(mouse[0]) * 0.01
    elev = 1.28 + float(mouse[1]) * 0.01                 # near face-on (down the axis)
    dist = 6.0
    eye = (math.sin(az) * dist * math.cos(elev), math.sin(elev) * dist,
           math.cos(az) * dist * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=52.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(gwaves_kernel, dim=(height, width),
              inputs=[img, cam, float(freq), float(omega), float(amp), float(orbit_r),
                      float(spin), float(ns_glow), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    if prog > 0.97:                                      # merge flash
        f = (prog - 0.97) / 0.03
        hdr = hdr + np.array([0.8, 0.88, 1.15], np.float32) * (f * 2.5)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)
