"""A beating heart — the cardiac cycle, lub-dub.

A heart-shaped body (two lobes + a downward tip) that **contracts** (systole) and
**relaxes** (diastole) about once a second, glowing with each beat — the wave of
excitation from the sinoatrial node driving the pump. Ray-marched SDF. See
``docs/research/24-the-living-body.md``. iMouse orbits; --frames animates.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _smin(a: float, b: float, k: float) -> float:
    h = wp.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b * (1.0 - h) + a * h - k * h * (1.0 - h)


@wp.func
def _heart(p: wp.vec3) -> float:
    s1 = wp.length(p - wp.vec3(-0.42, 0.35, 0.0)) - 0.55
    s2 = wp.length(p - wp.vec3(0.42, 0.35, 0.0)) - 0.55
    lobes = _smin(s1, s2, 0.25)
    c = 0.7071
    q = p - wp.vec3(0.0, -0.05, 0.0)
    rx = q[0] * c - q[1] * c
    ry = q[0] * c + q[1] * c
    qq = wp.vec3(wp.abs(rx) - 0.5, wp.abs(ry) - 0.5, wp.abs(q[2]) - 0.4)
    box = wp.length(wp.vec3(wp.max(qq[0], 0.0), wp.max(qq[1], 0.0), wp.max(qq[2], 0.0))) \
        + wp.min(wp.max(qq[0], wp.max(qq[1], qq[2])), 0.0) - 0.05
    return _smin(lobes, box, 0.22)


@wp.func
def _map(p: wp.vec3, scale: float) -> float:
    return _heart(p / scale) * scale


@wp.func
def _nrm(p: wp.vec3, scale: float) -> wp.vec3:
    e = 0.012
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e, 0.0, 0.0), scale) - _map(p - wp.vec3(e, 0.0, 0.0), scale),
        _map(p + wp.vec3(0.0, e, 0.0), scale) - _map(p - wp.vec3(0.0, e, 0.0), scale),
        _map(p + wp.vec3(0.0, 0.0, e), scale) - _map(p - wp.vec3(0.0, 0.0, e), scale)))


@wp.kernel
def heart_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, scale: float,
                 glow: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    bg = wp.vec3(0.06, 0.01, 0.02) * (0.6 + 0.4 * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(96):
        p = ro + rd * t
        d = _map(p, scale)
        if d < 0.002:
            hit = 1
            break
        t = t + d
        if t > 12.0:
            break
    col = bg
    if hit == 1:
        p = ro + rd * t
        n = _nrm(p, scale)
        vein = 0.85 + 0.3 * fbm3(p * 5.0, 3)           # muscle / vessel texture
        base = wp.vec3(0.75, 0.12, 0.14) * vein
        key = wp.normalize(wp.vec3(0.5, 0.7, 0.6))
        ndl = wp.max(wp.dot(n, key), 0.0)
        h = wp.normalize(key - rd)
        spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 30.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        col = base * (0.2 + 0.85 * ndl) + wp.vec3(1.0, 0.7, 0.6) * (spec * 0.5)
        col = col + wp.vec3(1.0, 0.2, 0.2) * (fres * 0.5)
        col = col + wp.vec3(1.0, 0.3, 0.25) * (glow * (0.3 + 0.7 * fres))   # beat throb
    img[i, j] = col


def _render(width, height, time, mouse, device, period=1.1):
    bt = (time % period) / period
    # lub-dub: two quick contractions
    lub = math.exp(-((bt - 0.05) ** 2) / 0.004)
    dub = 0.6 * math.exp(-((bt - 0.22) ** 2) / 0.004)
    pulse = min(1.0, lub + dub)
    scale = 1.0 - 0.11 * pulse                          # systole contracts
    glow = pulse
    cam = orbit_camera(width, height, time, mouse, dist=4.2, fov=44.0, el0=0.18,
                       auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(heart_kernel, dim=(height, width),
              inputs=[img, cam, float(scale), float(glow), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="heartbeat",
    description="A beating heart — a heart-shaped body contracting (systole) and "
                "relaxing (diastole) on a ~1 Hz lub-dub, glowing with each beat. "
                "iMouse orbits; --frames animates the beat.",
    renderer=_render,
)
