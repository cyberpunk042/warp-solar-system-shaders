"""A crystal cave — glowing gem shards in the dark.

A cluster of translucent crystal shards rising from a cave floor, each glowing
from within in amethyst, cyan and teal, lighting the dark rock around them and
hazing the air with coloured light. The crystals are sphere-traced octahedral
SDFs; a proximity halo accumulated along each ray gives the volumetric glow and
god-rays. --frames drifts the camera.

A new subject built on the engine's SDF raymarch — no existing scene touched.
"""

import math

import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.func
def _octa(p: wp.vec3, s: float) -> float:
    q = wp.vec3(wp.abs(p[0]), wp.abs(p[1]), wp.abs(p[2]))
    return (q[0] + q[1] + q[2] - s) * 0.57735


@wp.func
def _shard(p: wp.vec3, cx: float, cy: float, cz: float, s: float, yk: float) -> float:
    q = wp.vec3(p[0] - cx, (p[1] - cy) * yk, p[2] - cz)
    return _octa(q, s) / yk                            # elongated in y (tall crystal)


@wp.func
def _tint(idv: float) -> wp.vec3:
    if idv < 1.5:
        return wp.vec3(0.62, 0.30, 0.95)               # amethyst
    if idv < 2.5:
        return wp.vec3(0.20, 0.80, 0.95)               # cyan
    if idv < 3.5:
        return wp.vec3(0.30, 0.92, 0.60)               # teal-green
    return wp.vec3(0.95, 0.42, 0.72)                   # pink


@wp.func
def _map(p: wp.vec3) -> wp.vec2:
    """Return (distance, material id). id 0 = rock floor, 1..4 = crystals."""
    d = p[1] + 1.35 - 0.18 * fbm3(p * 0.8, 3)          # bumpy cave floor
    idv = float(0.0)
    s = _shard(p, 0.0, -0.5, 0.0, 0.62, 0.30)
    if s < d:
        d = s
        idv = 1.0
    s = _shard(p, -0.95, -0.7, 0.35, 0.42, 0.34)
    if s < d:
        d = s
        idv = 2.0
    s = _shard(p, 0.85, -0.75, -0.25, 0.5, 0.32)
    if s < d:
        d = s
        idv = 3.0
    s = _shard(p, 0.25, -0.85, 0.9, 0.34, 0.36)
    if s < d:
        d = s
        idv = 4.0
    s = _shard(p, -0.5, -0.9, -0.7, 0.3, 0.4)
    if s < d:
        d = s
        idv = 1.0
    return wp.vec2(d, idv)


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.004
    dx = _map(p + wp.vec3(e, 0.0, 0.0))[0] - _map(p - wp.vec3(e, 0.0, 0.0))[0]
    dy = _map(p + wp.vec3(0.0, e, 0.0))[0] - _map(p - wp.vec3(0.0, e, 0.0))[0]
    dz = _map(p + wp.vec3(0.0, 0.0, e))[0] - _map(p - wp.vec3(0.0, 0.0, e))[0]
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                  steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.0)
    hit = int(0)
    idv = float(0.0)
    glow = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(steps):
        p = ro + rd * t
        m = _map(p)
        d = m[0]
        # coloured proximity halo → volumetric glow + god-rays
        glow = glow + _tint(m[1]) * (wp.exp(-d * 14.0) * 0.016)
        if d < 0.001:
            hit = 1
            idv = m[1]
            break
        t += wp.max(d * 0.85, 0.004)
        if t > 20.0:
            break

    col = wp.vec3(0.015, 0.014, 0.022)                 # dark cave
    if hit == 1:
        p = ro + rd * t
        n = _normal(p)
        if idv < 0.5:
            # rock floor, lit only by nearby crystal glow (carried in `glow`)
            col = wp.vec3(0.05, 0.045, 0.06) * (0.4 + 0.6 * wp.max(n[1], 0.0))
        else:
            tint = _tint(idv)
            fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
            facet = 0.5 + 0.5 * wp.sin(p[0] * 9.0 + p[1] * 9.0 + p[2] * 9.0)
            # emissive interior + bright fresnel edges + faceted sparkle
            col = tint * (0.6 + 0.8 * facet) + wp.vec3(1.0, 1.0, 1.0) * (fres * 0.7)

    col = col + glow
    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.6 + float(mouse[0]) * 0.008 + time * 0.05
    el = 0.18 + float(mouse[1]) * 0.004
    dist = 4.2
    eye = (dist * math.cos(el) * math.sin(az), 0.3 + dist * math.sin(el),
           dist * math.cos(el) * math.cos(az))
    ss = 2
    W, H = int(width) * ss, int(height) * ss
    cam = make_camera(eye, (0.0, -0.2, 0.0), fov_deg=52.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, float(time), int(140), int(W), int(H)], device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss)
    r = max(3, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.6, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=0.92, preserve_hue=True)


SCENE = Scene(
    name="crystal_cave",
    description="A cluster of translucent crystal shards glowing from within in "
                "amethyst, cyan and teal, rising from a dark cave floor and hazing "
                "the air with coloured light and god-rays. --frames drifts the camera.",
    renderer=_render,
)
