"""A coral reef — the ocean's most crowded city.

Where sunlight reaches, colonial polyps build calcium-carbonate skeletons — branching,
brain and fan corals — in symbiosis with photosynthetic algae, sheltering a quarter of
all marine species. Sun shafts filter through blue water onto a reef bed of coral
mounds, fish darting between them. See ``docs/research/28-the-deep-ocean.md``.
--frames drifts the fish and the light.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.field import sd_capsule
from ..subatomic.render import orbit_camera
from ..scene import Scene

_C1 = wp.constant(wp.vec3(-0.72, -0.55, 0.25))     # brain coral
_C2 = wp.constant(wp.vec3(0.75, -0.75, -0.15))     # branching coral base
_C3 = wp.constant(wp.vec3(0.15, -0.7, 0.7))        # fan / second cluster


@wp.func
def _branch(p: wp.vec3, base: wp.vec3) -> float:
    d = sd_capsule(p, base, base + wp.vec3(0.0, 0.5, 0.0), 0.07)
    d = wp.min(d, sd_capsule(p, base + wp.vec3(0.0, 0.25, 0.0),
                             base + wp.vec3(0.28, 0.6, 0.1), 0.05))
    d = wp.min(d, sd_capsule(p, base + wp.vec3(0.0, 0.25, 0.0),
                             base + wp.vec3(-0.22, 0.62, -0.12), 0.05))
    d = wp.min(d, sd_capsule(p, base + wp.vec3(0.0, 0.4, 0.0),
                             base + wp.vec3(0.05, 0.8, -0.2), 0.04))
    return d


@wp.func
def _map(p: wp.vec3) -> float:
    floor = p[1] + 1.0 - 0.16 * fbm3(wp.vec3(p[0] * 1.1, 0.0, p[2] * 1.1), 4) \
        - 0.05 * fbm3(wp.vec3(p[0] * 4.0, 1.0, p[2] * 4.0), 2)
    brain = wp.length(p - _C1) - (0.5 + 0.05 * wp.sin(p[0] * 16.0) * wp.sin(p[2] * 16.0)
                                  * wp.sin(p[1] * 16.0))
    br = _branch(p, _C2)
    br2 = _branch(p, _C3 + wp.vec3(0.0, -0.05, 0.0))
    return wp.min(wp.min(floor, brain), wp.min(br, br2))


@wp.func
def _shade_col(p: wp.vec3) -> wp.vec3:
    floor = p[1] + 1.0 - 0.16 * fbm3(wp.vec3(p[0] * 1.1, 0.0, p[2] * 1.1), 4) \
        - 0.05 * fbm3(wp.vec3(p[0] * 4.0, 1.0, p[2] * 4.0), 2)
    brain = wp.length(p - _C1) - 0.5
    br = _branch(p, _C2)
    br2 = _branch(p, _C3 + wp.vec3(0.0, -0.05, 0.0))
    m = wp.min(wp.min(floor, brain), wp.min(br, br2))
    if m == floor:
        sand = 0.5 + 0.5 * fbm3(wp.vec3(p[0] * 5.0, 0.0, p[2] * 5.0), 3)
        return wp.vec3(0.55, 0.5, 0.4) * (0.5 + 0.5 * sand)
    if m == brain:
        return wp.vec3(0.95, 0.55, 0.35)                 # orange brain coral
    if m == br:
        return wp.vec3(0.95, 0.35, 0.6)                  # pink branching
    return wp.vec3(0.55, 0.4, 0.9)                       # purple cluster


@wp.kernel
def reef_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.3)
    hitp = wp.vec3(0.0, 0.0, 0.0)
    hit = int(0)
    for _ in range(120):
        p = ro + rd * t
        d = _map(p)
        if d < 0.002:
            hitp = p
            hit = 1
            break
        t += wp.max(d * 0.8, 0.01)
        if t > 12.0:
            break

    # water colour deepens (bluer, darker) with depth and distance
    up = wp.max(rd[1], 0.0)
    water = wp.vec3(0.015, 0.13, 0.24) + wp.vec3(0.03, 0.20, 0.28) * up
    if hit == 1:
        e = 0.006
        n = wp.normalize(wp.vec3(
            _map(hitp + wp.vec3(e, 0.0, 0.0)) - _map(hitp - wp.vec3(e, 0.0, 0.0)),
            _map(hitp + wp.vec3(0.0, e, 0.0)) - _map(hitp - wp.vec3(0.0, e, 0.0)),
            _map(hitp + wp.vec3(0.0, 0.0, e)) - _map(hitp - wp.vec3(0.0, 0.0, e))))
        sun = wp.normalize(wp.vec3(0.25, 1.0, 0.3))
        dif = wp.max(wp.dot(n, sun), 0.0)
        amb = 0.3 + 0.2 * n[1]
        base = _shade_col(hitp)
        col = base * (amb + 0.9 * dif)
        # caustic ripples brighten the lit tops
        caus = 0.5 + 0.5 * wp.sin(hitp[0] * 8.0 + time) * wp.sin(hitp[2] * 8.0 - time * 0.7)
        col = col + base * caus * dif * 0.3
        # blue water absorption with distance (mild, to keep corals vivid)
        fog = 1.0 - wp.exp(-t * 0.10)
        col = col * (1.0 - fog) + water * fog
    else:
        col = water

    # sun shafts + a few fish motes (screen additive along the ray)
    shaft = wp.pow(wp.max(rd[1], 0.0), 2.0) * (0.4 + 0.6 * fbm3(
        wp.vec3(rd[0] * 5.0 + time * 0.2, rd[1] * 3.0, rd[2] * 5.0), 3))
    col = col + wp.vec3(0.4, 0.7, 0.8) * shaft * 0.3
    for k in range(6):
        a = time * 0.5 + float(k) * 1.05
        fp = wp.vec3(0.9 * wp.sin(a * 1.3 + float(k)), -0.2 + 0.3 * wp.sin(a),
                     0.9 * wp.cos(a + float(k)))
        fd = wp.length(wp.cross(fp - ro, rd)) / wp.length(rd)
        # only draw fish in front of any surface hit
        along = wp.dot(fp - ro, rd)
        if along > 0.0 and (hit == 0 or along < t):
            col = col + wp.vec3(1.0, 0.85, 0.4) * wp.exp(-(fd / 0.03) * (fd / 0.03)) * 0.8

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=4.0, fov=46.0, el0=0.16,
                       auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(reef_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="coral_reef",
    description="A sunlit coral reef — a bed of branching, brain and fan corals in "
                "orange, pink and purple, sun shafts filtering through blue water, "
                "fish darting between the mounds. --frames drifts the fish and light.",
    renderer=_render,
)
