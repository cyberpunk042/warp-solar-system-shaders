"""A deep-sea jellyfish — a pulsing bell trailing glowing tentacles.

In the midnight zone the only light is **biological**. A translucent bell pulses by
jet propulsion, rim-lit and glowing a cold blue-green (~480 nm — the colour that
travels furthest in seawater), trailing a curtain of bioluminescent tentacles that
sway and spark at their tips. See ``docs/research/28-the-deep-ocean.md``.
--frames pulses the bell and drifts the tentacles.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..subatomic.field import void
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.kernel
def jelly_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, rx: float, ry: float,
                 time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 2.5)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 80.0
    t = t0 + dt * 0.5
    by = 0.45
    acc = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(80):
        p = ro + rd * t
        q = p - wp.vec3(0.0, by, 0.0)
        # translucent bell (upper ellipsoid shell)
        if q[1] > -0.25 * ry:
            e = wp.length(wp.vec3(q[0] / rx, q[1] / ry, q[2] / 0.9)) - 1.0
            shell = wp.exp(-(e / 0.07) * (e / 0.07))
            rim = wp.pow(wp.max(1.0 - wp.abs(q[1] / ry), 0.0), 1.5)
            acc = acc + wp.vec3(0.25, 0.7, 1.0) * shell * (0.5 + 0.8 * rim) * dt * 1.4
            acc = acc + wp.vec3(0.7, 0.95, 1.0) * shell * wp.pow(rim, 3.0) * dt * 0.8
            inner = wp.exp(-(e / 0.35) * (e / 0.35))          # faint filled dome
            acc = acc + wp.vec3(0.1, 0.35, 0.6) * inner * dt * 0.5
        # tentacles: 10 wavy glowing strands hanging from the rim
        tlen = 1.7
        d = (by - 0.06 - p[1]) / tlen
        if d > 0.0 and d < 1.0:
            for k in range(10):
                a = 6.2831 * float(k) / 10.0
                sway = 0.16 * wp.sin(d * 5.5 + time * 3.0 + float(k))
                tx = rx * 0.92 * wp.cos(a) + sway * wp.cos(a + 1.57)
                tz = 0.9 * 0.92 * wp.sin(a) + sway * wp.sin(a + 1.57)
                dist = wp.length(wp.vec2(p[0] - tx, p[2] - tz))
                gl = wp.exp(-(dist / 0.03) * (dist / 0.03))
                tip = wp.pow(d, 2.0)                            # brighter toward tips
                acc = acc + wp.vec3(0.9, 0.4, 0.95) * gl * (0.4 + 1.3 * tip) * dt * 1.1
        t += dt
    img[i, j] = acc * 1.5 + void(rd)


def _render(width, height, time, mouse, device):
    pulse = math.sin(time * 2.2)
    ry = 0.62 * (1.0 + 0.16 * pulse)
    rx = 0.92 * (1.0 - 0.10 * pulse)
    ss = 2
    W, H = int(width) * ss, int(height) * ss
    cam = orbit_camera(W, H, time, mouse, dist=4.2, fov=42.0, el0=0.12, auto=0.12)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(jelly_kernel, dim=(H, W),
              inputs=[img, cam, float(rx), float(ry), float(time), int(W), int(H)],
              device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy().astype(np.float32), ss)
    hdr += np.array([0.004, 0.010, 0.020], np.float32)      # deep-water base
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=0.8, strength=0.6, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.08, preserve_hue=True)


SCENE = Scene(
    name="jellyfish",
    description="A deep-sea jellyfish — a translucent bell pulsing by jet propulsion, "
                "rim-lit and glowing cold blue-green, trailing a curtain of "
                "bioluminescent tentacles that sway and spark at their tips. "
                "--frames pulses it.",
    renderer=_render,
)
