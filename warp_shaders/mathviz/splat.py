"""Additive point-splatting for math-visualisation scenes.

A shared projector + scatter kernel: given a cloud of 3-D points (a chaotic
trajectory, a knot tube, a 4-D projection) and per-point colours, orbit a pinhole
camera and additively splat each point (3×3, Gaussian-weighted) into an HDR image.
Reused by the strange-attractor, torus-knot, Klein-bottle and tesseract scenes.
"""

import math

import numpy as np
import warp as wp


@wp.kernel
def splat_kernel(pts: wp.array(dtype=wp.vec3), cols: wp.array(dtype=wp.vec3),
                 img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                 rgt: wp.vec3, upv: wp.vec3, foc: float, aspect: float,
                 intensity: float, width: int, height: int):
    k = wp.tid()
    d = pts[k] - eye
    cz = wp.dot(d, fwd)
    if cz < 0.05:
        return
    sx = wp.dot(d, rgt) / cz * foc
    sy = wp.dot(d, upv) / cz * foc
    px = (sx / aspect * 0.5 + 0.5) * float(width)
    py = (0.5 - sy * 0.5) * float(height)
    ci = int(py)
    cj = int(px)
    c = cols[k]
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            ii = ci + oy
            jj = cj + ox
            if ii >= 0 and ii < height and jj >= 0 and jj < width:
                w = wp.exp(-float(ox * ox + oy * oy) * 0.9)
                wp.atomic_add(img, ii, jj, c * (w * intensity))


def splat_scene(pts_np, cols_np, width, height, time, device, foc=1.9, dist=3.7,
                el=0.22, az0=0.6, az_speed=0.14, intensity=0.05,
                bg=(0.01, 0.014, 0.025)):
    """Orbit a camera around the origin and splat ``pts_np`` (N,3) coloured by
    ``cols_np`` (N,3) into an HDR float image. Returns the HDR array (pre-post)."""
    az = az0 + time * az_speed
    eye = wp.vec3(dist * math.cos(el) * math.cos(az), dist * math.sin(el),
                  dist * math.cos(el) * math.sin(az))
    fwd = wp.normalize(wp.vec3(-eye[0], -eye[1], -eye[2]))
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    pts = wp.array(pts_np.astype(np.float32), dtype=wp.vec3, device=device)
    cols = wp.array(cols_np.astype(np.float32), dtype=wp.vec3, device=device)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(splat_kernel, dim=pts_np.shape[0],
              inputs=[pts, cols, img, eye, fwd, rgt, upv, float(foc),
                      float(width / height), float(intensity), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr += np.array(bg, np.float32)
    return hdr
