"""A whale fall — an oasis on the abyssal plain.

When a whale dies and sinks to the deep seafloor, its carcass feeds a succession of
scavengers, bone-eating worms and glowing **bacterial mats** for decades — a burst of
life in the food-poor abyss. Here the skeleton rests on dark sediment, haloed by the
pale bloom of chemosynthetic bacteria and drifting scavenger sparks. See
``docs/research/28-the-deep-ocean.md``. --frames drifts the scavengers.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.field import sd_capsule
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _ellip(p: wp.vec3, c: wp.vec3, r: wp.vec3) -> float:
    q = wp.vec3((p[0] - c[0]) / r[0], (p[1] - c[1]) / r[1], (p[2] - c[2]) / r[2])
    return (wp.length(q) - 1.0) * wp.min(r[0], wp.min(r[1], r[2]))


@wp.func
def _bones(p: wp.vec3) -> float:
    # skull + spine + rib cage
    d = _ellip(p, wp.vec3(-1.45, -0.42, 0.0), wp.vec3(0.5, 0.3, 0.3))
    d = wp.min(d, sd_capsule(p, wp.vec3(-1.0, -0.5, 0.0), wp.vec3(1.5, -0.52, 0.0), 0.07))
    for k in range(7):
        tx = -0.75 + float(k) * 0.36
        arc = 0.6 - 0.03 * float((k - 3) * (k - 3))       # cage tapers at the ends
        d = wp.min(d, sd_capsule(p, wp.vec3(tx, -0.5, 0.0),
                                 wp.vec3(tx + 0.05, -0.5 + arc, arc), 0.032))
        d = wp.min(d, sd_capsule(p, wp.vec3(tx, -0.5, 0.0),
                                 wp.vec3(tx + 0.05, -0.5 + arc, -arc), 0.032))
    return d


@wp.func
def _map(p: wp.vec3) -> float:
    floor = p[1] + 0.72 - 0.05 * fbm3(wp.vec3(p[0] * 1.4, 0.0, p[2] * 1.4), 3)
    return wp.min(floor, _bones(p))


@wp.kernel
def whale_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                 width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.3)
    hitp = wp.vec3(0.0, 0.0, 0.0)
    hit = int(0)
    for _ in range(130):
        p = ro + rd * t
        d = _map(p)
        if d < 0.002:
            hitp = p
            hit = 1
            break
        t += wp.max(d * 0.8, 0.008)
        if t > 12.0:
            break

    col = wp.vec3(0.006, 0.012, 0.02)               # near-black water
    if hit == 1:
        e = 0.005
        n = wp.normalize(wp.vec3(
            _map(hitp + wp.vec3(e, 0.0, 0.0)) - _map(hitp - wp.vec3(e, 0.0, 0.0)),
            _map(hitp + wp.vec3(0.0, e, 0.0)) - _map(hitp - wp.vec3(0.0, e, 0.0)),
            _map(hitp + wp.vec3(0.0, 0.0, e)) - _map(hitp - wp.vec3(0.0, 0.0, e))))
        floor = hitp[1] + 0.72 - 0.05 * fbm3(wp.vec3(hitp[0] * 1.4, 0.0, hitp[2] * 1.4), 3)
        bone = _bones(hitp)
        amb = 0.25 + 0.3 * wp.max(n[1], 0.0)
        if bone < floor:                            # bone: pale, self-lit by bacteria
            band = 0.8 + 0.2 * wp.sin(hitp[0] * 40.0)
            col = wp.vec3(0.85, 0.82, 0.72) * (amb + 0.5) * band
        else:                                       # sediment + bacterial-mat bloom
            sed = wp.vec3(0.06, 0.055, 0.05) * amb
            # mat glows along the body footprint (near the spine axis)
            body = wp.exp(-(hitp[2] * hitp[2]) * 7.0) \
                * wp.smoothstep(1.9, 0.0, wp.abs(hitp[0]) - 0.2)
            mat = fbm3(wp.vec3(hitp[0] * 5.0, 1.0, hitp[2] * 5.0), 3)
            col = sed + wp.vec3(0.85, 0.95, 0.55) * body * (0.2 + 0.8 * mat) * 0.4

    # scavenger sparks + marine snow drifting above the fall
    for k in range(8):
        a = time * 0.4 + float(k) * 0.9
        fp = wp.vec3(-1.4 + 0.42 * float(k) + 0.2 * wp.sin(a * 1.7),
                     -0.35 + 0.28 * wp.abs(wp.sin(a)), 0.35 * wp.cos(a + float(k)))
        fd = wp.length(wp.cross(fp - ro, rd)) / wp.length(rd)
        along = wp.dot(fp - ro, rd)
        if along > 0.0 and (hit == 0 or along < t):
            col = col + wp.vec3(0.5, 0.95, 0.8) * wp.exp(-(fd / 0.02) * (fd / 0.02)) * 0.7

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=4.6, fov=44.0, el0=0.2,
                       auto=0.09)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(whale_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr += np.array([0.004, 0.008, 0.013], np.float32)
    r = max(2, int(min(width, height) * 0.011))
    hdr = post.bloom(hdr, threshold=0.85, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="whale_fall",
    description="A whale fall — a whale skeleton resting on the dark abyssal floor, "
                "haloed by the pale bloom of chemosynthetic bacterial mats and drifting "
                "scavenger sparks: an oasis of life in the food-poor deep. --frames "
                "drifts the scavengers.",
    renderer=_render,
)
