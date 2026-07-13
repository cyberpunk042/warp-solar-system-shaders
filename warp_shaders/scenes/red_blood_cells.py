"""Red blood cells — biconcave discs tumbling through the bloodstream.

Erythrocytes: dimpled biconcave discs packed with haemoglobin, carrying oxygen,
drifting through warm plasma. Domain-repeated SDF discs, each tilted, flowing past
the camera. See ``docs/research/24-the-living-body.md``. --frames animates the flow.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.hash import hash21
from ..subatomic.render import orbit_camera
from ..scene import Scene

_SP = 1.5                                               # cell spacing


@wp.func
def _rbc(q: wp.vec3) -> float:
    disc = wp.length(wp.cw_mul(q, wp.vec3(1.0, 2.3, 1.0))) - 0.44
    top = wp.length(q - wp.vec3(0.0, 0.42, 0.0)) - 0.42
    bot = wp.length(q - wp.vec3(0.0, -0.42, 0.0)) - 0.42
    d = wp.max(disc, -top)
    return wp.max(d, -bot)


@wp.func
def _map(p: wp.vec3) -> float:
    cx = wp.floor(p[0] / _SP + 0.5)
    cz = wp.floor(p[2] / _SP + 0.5)
    jit = hash21(wp.vec2(cx, cz))
    q = wp.vec3(p[0] - cx * _SP, p[1] - (jit - 0.5) * 1.2, p[2] - cz * _SP)
    # per-cell tilt about z then x
    a = (jit - 0.5) * 2.2
    c = wp.cos(a)
    s = wp.sin(a)
    q = wp.vec3(q[0] * c - q[1] * s, q[0] * s + q[1] * c, q[2])
    return _rbc(q)


@wp.func
def _nrm(p: wp.vec3) -> wp.vec3:
    e = 0.012
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0)),
        _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0)),
        _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))))


@wp.kernel
def rbc_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, flow: float,
               width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye + wp.vec3(flow, 0.0, 0.0)              # flow past the camera
    rd = camera_ray_dir(cam, u, v)
    # warm plasma background
    bg = wp.vec3(0.5, 0.13, 0.12) * (0.7 + 0.3 * v) + wp.vec3(0.2, 0.03, 0.03)

    t = float(0.0)
    hit = int(0)
    for _ in range(90):
        p = ro + rd * t
        d = _map(p)
        if d < 0.003:
            hit = 1
            break
        t = t + d * 0.85
        if t > 16.0:
            break
    col = bg
    if hit == 1:
        p = ro + rd * t
        n = _nrm(p)
        key = wp.normalize(wp.vec3(0.5, 0.7, 0.6))
        ndl = wp.max(wp.dot(n, key), 0.0)
        h = wp.normalize(key - rd)
        spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 24.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        base = wp.vec3(0.75, 0.13, 0.1)
        col = base * (0.22 + 0.85 * ndl) + wp.vec3(1.0, 0.6, 0.5) * (spec * 0.4)
        col = col + wp.vec3(1.0, 0.4, 0.35) * (fres * 0.45)
        col = col * (0.4 + 0.6 * wp.exp(-t * 0.06))     # depth fade into plasma
    img[i, j] = col


def _render(width, height, time, mouse, device):
    flow = time * 0.7
    cam = orbit_camera(width, height, time, mouse, dist=5.5, fov=48.0, el0=0.16,
                       auto=0.0)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(rbc_kernel, dim=(height, width),
              inputs=[img, cam, float(flow), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.4, strength=0.3, radius=r, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="red_blood_cells",
    description="Red blood cells — biconcave, dimpled erythrocytes tumbling through "
                "warm plasma, carrying oxygen. Domain-repeated SDF discs flowing past "
                "the camera. --frames animates the bloodstream.",
    renderer=_render,
)
