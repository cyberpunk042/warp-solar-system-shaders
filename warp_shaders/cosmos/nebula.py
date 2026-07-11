"""A positioned, sized volumetric nebula — reusable in the composited system.

`nebula_at` gives emission colour + density at a point (filamentary fBm/ridged
cloud inside a bounding sphere, cool violet-blue skirts warming to a bright
core); `nebula_march` integrates it front-to-back along a ray between the
nebula's bounding-sphere entry/exit, returning premultiplied emission +
transmittance so the system compositor can lay it behind the bodies. The centre
and radius are parameters, so a scene chooses where the nebula sits and how big.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, ridged3


@wp.func
def nebula_at(p: wp.vec3, center: wp.vec3, radius: float, seed: float,
              time: float) -> wp.vec4:
    d = p - center
    r = wp.length(d) / wp.max(radius, 1e-4)
    if r > 1.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)
    nd = d / wp.max(radius, 1e-4)
    s = wp.vec3(seed, seed * 1.7, seed * 2.3)
    w = time * 0.01
    base = fbm3(nd * 3.5 + s + wp.vec3(w, 0.0, 0.0), 5)
    fil = ridged3(nd * 6.0 + s + wp.vec3(w * 0.5, 0.0, 0.0), 4)
    # high threshold -> voids + bright filaments instead of a uniform fog
    dens = wp.clamp((base * 0.55 + fil * 0.62 - 0.56) * 1.9, 0.0, 1.0)
    dens = dens * wp.smoothstep(1.0, 0.3, r)              # fade to the edge
    if dens <= 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 0.0)               # empty step: skip colour ramps
    core = wp.smoothstep(0.62, 0.0, r)
    knot = wp.smoothstep(0.72, 0.96, fil)                # bright dense knots
    cool = wp.vec3(0.28, 0.16, 0.62)                     # violet-blue skirts
    warm = wp.vec3(0.95, 0.34, 0.5)                      # rose mid
    hot = wp.vec3(1.0, 0.84, 0.62)                       # hot knots / core
    col = cool * (1.0 - core) + warm * core
    col = col + hot * (knot * (0.5 + 0.9 * core))
    return wp.vec4(col[0], col[1], col[2], dens)


@wp.func
def nebula_march(ro: wp.vec3, rd: wp.vec3, center: wp.vec3, radius: float,
                 seed: float, time: float, steps: int) -> wp.vec4:
    """Integrate the nebula along the ray. Returns (emission_rgb, transmittance);
    transmittance 1 = nothing hit."""
    oc = ro - center
    b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - radius * radius
    disc = b * b - c
    if disc < 0.0:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)
    sq = wp.sqrt(disc)
    t0 = wp.max(-b - sq, 0.0)
    t1 = -b + sq
    if t1 <= t0:
        return wp.vec4(0.0, 0.0, 0.0, 1.0)
    seg = (t1 - t0) / float(steps)
    t = t0 + 0.5 * seg
    trans = float(1.0)
    acc = wp.vec3(0.0, 0.0, 0.0)
    # Adaptive marching: the high density threshold leaves big voids, so step
    # 2x through empty space and fall back to the fine step the moment a filament
    # appears. `steps` bounds the fine-step budget; the coarse pass covers the
    # rest of [t0, t1] in far fewer samples. Same look, ~2x fewer noise taps.
    for _ in range(2 * steps):
        if t > t1:
            break
        p = ro + rd * t
        nv = nebula_at(p, center, radius, seed, time)
        raw = nv[3]
        if raw > 0.003:
            dens = raw * seg * 1.6
            em = wp.vec3(nv[0], nv[1], nv[2])
            acc = acc + em * (dens * trans)
            trans = trans * (1.0 - wp.clamp(dens, 0.0, 1.0))
            if trans < 0.02:
                break
            t += seg                                 # fine step inside a filament
        else:
            t += seg * 2.0                           # skip through the void
    return wp.vec4(acc[0], acc[1], acc[2], trans)


@wp.kernel
def nebula_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
                  center: wp.vec3, radius: float, seed: float, time: float,
                  steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)
    bg = stars(rd)
    nv = nebula_march(ro, rd, center, radius, seed, time, steps)
    col = bg * nv[3] + wp.vec3(nv[0], nv[1], nv[2])
    img[i, j] = col


def render_nebula(center=(0.0, 0.0, 0.0), radius=3.0, seed=1.0, time=0.0,
                  width=480, height=360, device="cpu", dist=9.0, fov=45.0,
                  steps=64) -> np.ndarray:
    eye = (dist * 0.2, dist * 0.15, dist)
    cam = make_camera(eye, (0.0, 0.0, 0.0), fov_deg=fov, aspect=width / height)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(nebula_kernel, dim=(height, width),
              inputs=[img, cam, wp.vec3(center[0], center[1], center[2]),
                      float(radius), float(seed), float(time), int(steps),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=1.1, strength=0.4,
                     radius=max(3, int(min(width, height) * 0.02)), passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)
