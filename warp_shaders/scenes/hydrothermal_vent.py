"""A hydrothermal vent — a black smoker on the abyssal floor.

At mid-ocean ridges seawater is superheated by magma and erupts back as mineral-laden
plumes: a **black smoker** billows iron-sulphide particles at ~350 °C into 4 °C water.
Around it thrives an ecosystem powered not by sunlight but by **chemosynthesis** —
giant red-plumed **tube worms** clustered on the chimney. See
``docs/research/28-the-deep-ocean.md``. --frames boils the plume.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.field import sd_capsule, void
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _surf(p: wp.vec3) -> float:
    floor = p[1] + 1.2 - 0.06 * fbm3(wp.vec3(p[0] * 1.5, 0.0, p[2] * 1.5), 3)
    chim = sd_capsule(p, wp.vec3(0.0, -1.25, 0.0), wp.vec3(0.0, 0.12, 0.0), 0.24)
    return wp.min(floor, chim)


@wp.kernel
def vent_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    t = float(0.5)
    dt = float(0.055)
    acc = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    col = void(rd) * 0.5
    hit = int(0)
    for _ in range(150):
        p = ro + rd * t
        # volumetric black-smoker plume (rising cone above the vent mouth)
        if p[1] > 0.08 and trans > 0.02:
            hh = p[1] - 0.08
            sway = 0.14 * wp.sin(hh * 2.2 + time * 1.5)
            rad = 0.17 + hh * 0.42
            rr = wp.length(wp.vec2(p[0] - sway, p[2]))
            cone = wp.smoothstep(rad, rad * 0.2, rr)
            turb = fbm3(wp.vec3(p[0] * 2.6, p[1] * 2.2 - time * 1.4, p[2] * 2.6), 4)
            dens = cone * (0.3 + 1.0 * turb) * wp.exp(-hh * 0.5)
            if dens > 0.0:
                hot = wp.exp(-hh * 3.2)
                ecol = wp.vec3(0.04, 0.04, 0.05) + wp.vec3(1.5, 0.5, 0.12) * hot
                acc = acc + ecol * (dens * trans * dt * 2.0)
                trans = trans * wp.exp(-dens * dt * 4.0)
        # tube worms: red-tipped white tubes ringing the chimney
        for k in range(7):
            a = 6.2831 * float(k) / 7.0
            wx = 0.34 * wp.cos(a)
            wz = 0.34 * wp.sin(a)
            dw = sd_capsule(p, wp.vec3(wx, -1.2, wz), wp.vec3(wx * 1.15, -0.5, wz * 1.15), 0.02)
            wg = wp.exp(-(dw / 0.03) * (dw / 0.03))
            acc = acc + wp.vec3(0.9, 0.15, 0.12) * wg * trans * dt * 1.4
        # opaque surfaces (seafloor + chimney)
        if _surf(p) < 0.0:
            n = wp.normalize(wp.vec3(
                _surf(p + wp.vec3(0.01, 0.0, 0.0)) - _surf(p - wp.vec3(0.01, 0.0, 0.0)),
                _surf(p + wp.vec3(0.0, 0.01, 0.0)) - _surf(p - wp.vec3(0.0, 0.01, 0.0)),
                _surf(p + wp.vec3(0.0, 0.0, 0.01)) - _surf(p - wp.vec3(0.0, 0.0, 0.01))))
            rr = wp.length(wp.vec2(p[0], p[2]))
            chim = sd_capsule(p, wp.vec3(0.0, -1.25, 0.0), wp.vec3(0.0, 0.12, 0.0), 0.24)
            floor = p[1] + 1.2 - 0.06 * fbm3(wp.vec3(p[0] * 1.5, 0.0, p[2] * 1.5), 3)
            sh = wp.vec3(0.0, 0.0, 0.0)
            if chim < floor:                        # the chimney: dark mineral rock
                crust = 0.5 + 0.5 * fbm3(wp.vec3(p[0] * 6.0, p[1] * 6.0, p[2] * 6.0), 3)
                rock = wp.vec3(0.05, 0.045, 0.05) * (0.5 + 0.7 * crust)
                heat = wp.clamp((p[1] + 0.1) / 0.28, 0.0, 1.0)   # hot only near the mouth
                sh = rock + wp.vec3(1.2, 0.35, 0.08) * heat * heat * 0.6
            else:                                   # seafloor: dark, red vent glow
                rock = wp.vec3(0.05, 0.05, 0.06) * (0.4 + 0.6 * wp.max(n[1], 0.0))
                sh = rock + wp.vec3(1.0, 0.32, 0.08) * wp.exp(-rr * rr * 1.7) * 0.8
            col = sh * trans + acc
            hit = 1
            break
        t += dt
        if t > 11.0:
            break
    if hit == 0:
        col = col * trans + acc
    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=4.6, fov=44.0, el0=0.16,
                       auto=0.1)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(vent_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr += np.array([0.004, 0.008, 0.014], np.float32)
    r = max(2, int(min(width, height) * 0.011))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.06)


SCENE = Scene(
    name="hydrothermal_vent",
    description="A black smoker on the abyssal floor — a mineral chimney billowing a "
                "hot iron-sulphide plume into near-freezing water, ringed by red-plumed "
                "tube worms living on chemosynthesis, not sunlight. --frames boils the "
                "plume.",
    renderer=_render,
)
