"""Gravitationally-lensed black hole — the strong-gravity body of the system.

Light is bent by integrating the GR photon-orbit equation

    d²x/dλ² = -3/2 · h² · x / |x|⁵          (h = x × v, conserved)

per pixel (this is the standard weak-to-strong deflection ODE; it conserves the
specific angular momentum h and produces the correct Einstein ring + photon
ring for free). Rays that fall inside the Schwarzschild radius are captured (the
black event-horizon disk); the rest sample the lensed background. A hot
accretion disk in the equatorial plane is crossed and accumulated along the bent
ray, with a temperature gradient (inner blue-white → outer orange) and
**Doppler beaming** — the side orbiting toward us is brighter and bluer.

`bh_pixel` is the reusable per-ray integrator (the composited system renderer
calls it, sampling its own background for the escaped ray); `render_black_hole`
renders one centred hole over the starfield for standalone verification.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3

from .bodies import StarConfig, make_star, BLACK_HOLE

_STEPS = 220


@wp.func
def _disk_emission(cp: wp.vec3, v: wp.vec3, rs: float, r_in: float,
                   r_out: float, time: float, spin: float) -> wp.vec3:
    r = wp.sqrt(cp[0] * cp[0] + cp[2] * cp[2])
    t = wp.clamp((r - r_in) / (r_out - r_in), 0.0, 1.0)
    inner = wp.vec3(0.95, 0.97, 1.05)                    # hot inner (blue-white)
    mid = wp.vec3(1.0, 0.85, 0.55)
    outer = wp.vec3(1.0, 0.42, 0.12)                     # cool outer (orange)
    if t < 0.5:
        col = inner * (1.0 - t / 0.5) + mid * (t / 0.5)
    else:
        col = mid * (1.0 - (t - 0.5) / 0.5) + outer * ((t - 0.5) / 0.5)
    # turbulent orbiting streaks (sheared by radius, drifting in time)
    ang = wp.atan2(cp[2], cp[0])
    q = wp.vec3(wp.cos(ang * 4.0) * r, ang * 2.0 + time * spin, wp.sin(ang * 4.0) * r)
    streak = fbm3(q * 0.8, 4)
    bright = (2.4 - 1.8 * t) * (0.45 + 0.9 * streak)     # inner disk far brighter
    # Doppler beaming: the orbital tangent vs the photon direction
    tang = wp.normalize(wp.vec3(-cp[2], 0.0, cp[0]))
    dop = wp.dot(tang, wp.normalize(v))
    beam = wp.clamp(1.0 + 1.6 * dop, 0.15, 3.2)
    shift = wp.vec3(1.0 - 0.15 * dop, 1.0, 1.0 + 0.18 * dop)   # blue approaching
    return wp.cw_mul(col, shift) * (bright * beam)


@wp.func
def bh_pixel(ro: wp.vec3, rd: wp.vec3, rs: float, time: float,
             spin: float) -> wp.vec4:
    """March one photon. Returns (disk_rgb, transmittance) — the escaped ray
    direction is re-derived by the caller; here we fold the lensed starfield in
    via `captured`: alpha channel <0 means captured (paint black)."""
    p = ro
    v = rd
    r_in = 2.3 * rs
    r_out = 6.5 * rs
    disk = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    captured = int(0)
    escaped = int(0)
    for _ in range(_STEPS):
        h = wp.cross(p, v)
        h2 = wp.dot(h, h)
        r2 = wp.dot(p, p)
        acc = p * (-1.5 * h2 / wp.pow(r2, 2.5))
        # step size grows with distance (fine near the hole, coarse far away)
        dt = 0.10 + 0.04 * wp.sqrt(r2)
        v = v + acc * dt
        pn = p + v * dt
        rn = wp.length(pn)
        if rn < rs:
            captured = 1
            break
        # accretion-disk crossing (equatorial plane y=0)
        if (p[1] * pn[1]) < 0.0:
            k = p[1] / (p[1] - pn[1])
            cp = p + (pn - p) * k
            rr = wp.sqrt(cp[0] * cp[0] + cp[2] * cp[2])
            if rr > r_in and rr < r_out:
                em = _disk_emission(cp, v, rs, r_in, r_out, time, spin)
                fade = wp.smoothstep(r_out, r_out * 0.6, rr)   # soft outer edge
                a = wp.clamp(0.5 + 0.5 * fade, 0.15, 1.0)
                disk = disk + em * (trans * a)
                trans = trans * (1.0 - a)
        p = pn
        if wp.length(p) > 60.0 * rs:
            escaped = 1
            break
    # pack: store escaped ray dir is lost; signal via w (>=0 escaped, <0 captured)
    w = trans
    if captured == 1:
        w = -1.0
    return wp.vec4(disk[0], disk[1], disk[2], w)


@wp.func
def bh_escape_dir(ro: wp.vec3, rd: wp.vec3, rs: float) -> wp.vec3:
    """The lensed (bent) direction a ray leaves with — for sampling background."""
    p = ro
    v = rd
    for _ in range(_STEPS):
        h = wp.cross(p, v)
        h2 = wp.dot(h, h)
        r2 = wp.dot(p, p)
        acc = p * (-1.5 * h2 / wp.pow(r2, 2.5))
        dt = 0.10 + 0.04 * wp.sqrt(r2)
        v = v + acc * dt
        p = p + v * dt
        if wp.length(p) < rs:
            return v
        if wp.length(p) > 60.0 * rs:
            break
    return wp.normalize(v)


@wp.kernel
def bh_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, cfg: StarConfig,
              time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)
    rs = cfg.radius

    dv = bh_pixel(ro, rd, rs, time, cfg.spin)
    disk = wp.vec3(dv[0], dv[1], dv[2])
    w = dv[3]
    if w < 0.0:
        col = disk                                       # captured: only disk glow
    else:
        bent = bh_escape_dir(ro, rd, rs)
        col = stars(bent) * w + disk                     # lensed starfield + disk
    img[i, j] = col


def render_black_hole(cfg: StarConfig, width: int, height: int,
                      time: float = 0.0, device: str = "cpu",
                      dist: float = 12.0, fov: float = 34.0,
                      elev: float = 0.16) -> np.ndarray:
    """Render one centred black hole (rs = cfg.radius) to an ``(H, W, 3)`` image."""
    eye = (dist * math.sin(0.5) * math.cos(elev), dist * math.sin(elev),
           dist * math.cos(0.5) * math.cos(elev))
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=fov, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bh_kernel, dim=(height, width),
              inputs=[img, cam, cfg, float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.5, radius=r, passes=4)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


def make_black_hole(radius=1.0, activity=1.0, spin=1.0, seed=1.0) -> StarConfig:
    return make_star(kind=BLACK_HOLE, radius=radius, temp=0.5, activity=activity,
                     spin=spin, seed=seed)
