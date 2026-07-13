"""A strange attractor — the shape of deterministic chaos.

The **Lorenz attractor** (1963): integrate ẋ=σ(y−x), ẏ=x(ρ−z)−y, ż=xy−βz and the
trajectory never repeats yet is trapped forever on a fractal butterfly — the
*butterfly effect* made visible. A swarm of points is integrated on the host and
its glowing path splatted additively. See ``docs/research/27-mathematics-made-visible.md``.
--frames orbits the attractor.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


def _lorenz(n=60000, dt=0.004, sig=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = 0.1, 0.0, 0.0
    for _ in range(1500):                         # warm-up onto the attractor
        dx = sig * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
    pts = np.empty((n, 3), np.float32)
    for k in range(n):
        dx = sig * (y - x); dy = x * (rho - z) - y; dz = x * y - beta * z
        x += dx * dt; y += dy * dt; z += dz * dt
        pts[k] = (x, y, z - 25.0)                 # centre on z
    pts *= 0.055
    return pts


_PTS = _lorenz()


@wp.kernel
def splat_kernel(pts: wp.array(dtype=wp.vec3), img: wp.array2d(dtype=wp.vec3),
                 eye: wp.vec3, fwd: wp.vec3, rgt: wp.vec3, upv: wp.vec3,
                 foc: float, aspect: float, n: int, width: int, height: int):
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
    f = float(k) / float(n)
    col = wp.vec3(0.35 + 0.65 * f, 0.55 + 0.35 * wp.sin(f * 9.0), 1.0 - 0.45 * f)
    for oy in range(-1, 2):
        for ox in range(-1, 2):
            ii = ci + oy
            jj = cj + ox
            if ii >= 0 and ii < height and jj >= 0 and jj < width:
                w = wp.exp(-float(ox * ox + oy * oy) * 0.9)
                wp.atomic_add(img, ii, jj, col * (w * 0.05))


def _render(width, height, time, mouse, device):
    az = 0.6 + time * 0.14
    el = 0.22
    dist = 3.7
    eye = wp.vec3(dist * math.cos(el) * math.cos(az), dist * math.sin(el),
                  dist * math.cos(el) * math.sin(az))
    fwd = wp.normalize(wp.vec3(-eye[0], -eye[1], -eye[2]))
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    pts = wp.array(_PTS, dtype=wp.vec3, device=device)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(splat_kernel, dim=_PTS.shape[0],
              inputs=[pts, img, eye, fwd, rgt, upv, 1.95, float(width / height),
                      _PTS.shape[0], int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    # faint deep-space background
    hdr += np.array([0.01, 0.014, 0.025], np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.6, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.1)


SCENE = Scene(
    name="strange_attractor",
    description="The Lorenz attractor — deterministic chaos drawn as a glowing fractal "
                "butterfly the trajectory is trapped on forever (the butterfly effect). "
                "A 60k-point swarm integrated on the host and splatted additively. "
                "--frames orbits it.",
    renderer=_render,
)
