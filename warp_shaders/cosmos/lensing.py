"""In-system black-hole lensing — a screen-space pass over the rendered scene.

The system is rendered first (stars, planets, nebula, starfield) into an image.
For each black hole this pass, per pixel, integrates the photon-orbit ODE in the
hole's frame (`blackhole.bh_pixel` / `bh_escape_dir`): captured rays paint the
black event horizon, and escaped rays sample the **already-rendered scene** in
their bent direction — so the whole system (its stars and planet included) warps
around the hole into an Einstein ring, and the Doppler-beamed accretion disk is
composited on top. Screen-space is an approximation (it lenses the flat image,
not true depth), but it gives the iconic effect cheaply and composites over any
scene the rest of the pipeline produced.
"""

from __future__ import annotations

import numpy as np
import warp as wp

from ..earthgfx import stars
from ..engine.uniforms import Camera, camera_ray_dir
from .blackhole import bh_escape_dir, bh_pixel


@wp.kernel
def _lens_kernel(out: wp.array2d(dtype=wp.vec3), scene: wp.array2d(dtype=wp.vec3),
                 cam: Camera, center: wp.vec3, rs: float, spin: float,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    vv = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, vv)
    rob = ro - center                                    # hole-centred frame

    dv = bh_pixel(rob, rd, rs, time, spin)
    disk = wp.vec3(dv[0], dv[1], dv[2])
    w = dv[3]
    if w < 0.0:
        out[i, j] = disk                                 # captured -> horizon + disk
        return
    bent = bh_escape_dir(rob, rd, rs)
    # sample the already-rendered scene in the bent direction: project a far
    # point along `bent` from the hole back onto the screen
    pf = center + bent * (200.0 * rs)
    rel = pf - ro
    z = wp.dot(rel, cam.forward)
    col = stars(bent)                                    # fallback if off-screen
    if z > 0.0:
        su = (wp.dot(rel, cam.right) / z) / (cam.aspect * cam.tan_half_fov)
        sv = (wp.dot(rel, cam.up) / z) / cam.tan_half_fov
        if su > -1.0 and su < 1.0 and sv > -1.0 and sv < 1.0:
            sj = int((su + 1.0) * 0.5 * float(width))
            si = int((float(height) - (sv + 1.0) * 0.5 * float(height)))
            if si >= 0 and si < height and sj >= 0 and sj < width:
                col = scene[si, sj]
    out[i, j] = col * w + disk


def apply_black_hole(scene: np.ndarray, cam: Camera, center, rs: float,
                     spin: float, time: float, device: str = "cpu") -> np.ndarray:
    """Return `scene` lensed by a black hole at `center` (Schwarzschild radius
    `rs`). `cam` is the system camera used to render `scene`."""
    h, w = scene.shape[:2]
    src = wp.array(scene.astype(np.float32), dtype=wp.vec3, device=device)
    out = wp.zeros((h, w), dtype=wp.vec3, device=device)
    wp.launch(_lens_kernel, dim=(h, w),
              inputs=[out, src, cam,
                      wp.vec3(float(center[0]), float(center[1]), float(center[2])),
                      float(rs), float(spin), float(time), int(w), int(h)],
              device=device)
    wp.synchronize_device(device)
    return out.numpy()
