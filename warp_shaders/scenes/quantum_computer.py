"""Quantum computer — a qubit on the Bloch sphere.

A qubit is not 0 *or* 1 but a **superposition** of both — a point on the **Bloch
sphere**, |0⟩ at the north pole, |1⟩ at the south. Here a glowing wireframe sphere
with X/Y/Z axes and a **state vector** that precesses and nutates through
superposition, plus an entangled partner. See ``docs/research/26-the-machine.md``.
iMouse orbits; --frames animates the state.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..subatomic.field import sd_capsule, void
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _dot(p: wp.vec3, c: wp.vec3, s: float) -> float:
    d = wp.length(p - c)
    return wp.exp(-(d / s) * (d / s))


@wp.kernel
def bloch_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, tip: wp.vec3,
                 tip2: wp.vec3, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 1.6)
    if g[0] > 1.0e28 or g[1] < 0.0:
        img[i, j] = void(rd)
        return
    t0 = wp.max(g[0], 0.0)
    dt = (g[1] - t0) / 60.0
    t = t0 + dt * 0.5
    acc = wp.vec3(0.0, 0.0, 0.0)
    o = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(60):
        p = ro + rd * t
        r = wp.length(p)
        # sphere wireframe: near the unit shell AND a coordinate great circle
        shell = wp.exp(-((r - 1.0) / 0.02) * ((r - 1.0) / 0.02))
        ring = wp.exp(-(p[1] / 0.02) * (p[1] / 0.02)) \
            + wp.exp(-(p[0] / 0.02) * (p[0] / 0.02)) \
            + wp.exp(-(p[2] / 0.02) * (p[2] / 0.02))
        e = wp.vec3(0.35, 0.55, 0.9) * (shell * wp.min(ring, 1.0) * 0.9)
        e = e + wp.vec3(0.1, 0.2, 0.4) * (shell * 0.12)          # faint full shell
        # axes
        e = e + wp.vec3(1.0, 0.3, 0.3) * wp.exp(-(sd_capsule(p, wp.vec3(-1.25, 0.0, 0.0), wp.vec3(1.25, 0.0, 0.0), 0.0) / 0.01) ** 2.0) * 0.4
        e = e + wp.vec3(0.3, 1.0, 0.4) * wp.exp(-(sd_capsule(p, wp.vec3(0.0, 0.0, -1.25), wp.vec3(0.0, 0.0, 1.25), 0.0) / 0.01) ** 2.0) * 0.4
        e = e + wp.vec3(0.5, 0.6, 1.0) * wp.exp(-(sd_capsule(p, wp.vec3(0.0, -1.25, 0.0), wp.vec3(0.0, 1.25, 0.0), 0.0) / 0.01) ** 2.0) * 0.4
        # the state vector (bright) + its entangled partner
        dv = sd_capsule(p, o, tip, 0.0)
        e = e + wp.vec3(1.0, 0.9, 0.4) * wp.exp(-(dv / 0.022) ** 2.0) * 2.0
        dv2 = sd_capsule(p, o, tip2, 0.0)
        e = e + wp.vec3(0.5, 0.9, 1.0) * wp.exp(-(dv2 / 0.02) ** 2.0) * 1.2
        # tips + poles
        e = e + wp.vec3(1.0, 0.95, 0.6) * _dot(p, tip, 0.06) * 2.0
        e = e + wp.vec3(0.9, 0.95, 1.0) * (_dot(p, wp.vec3(0.0, 1.0, 0.0), 0.05) + _dot(p, wp.vec3(0.0, -1.0, 0.0), 0.05)) * 1.2
        acc = acc + e * dt
        t += dt
    img[i, j] = acc * 2.2 + void(rd)


def _render(width, height, time, mouse, device):
    th = 0.9 + 0.5 * math.sin(time * 0.5)
    ph = time * 1.3
    tip = wp.vec3(math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph))
    th2 = math.pi - th
    ph2 = ph + math.pi
    tip2 = wp.vec3(math.sin(th2) * math.cos(ph2) * 0.9, math.cos(th2) * 0.9, math.sin(th2) * math.sin(ph2) * 0.9)
    cam = orbit_camera(width, height, time, mouse, dist=4.0, fov=42.0, el0=0.25,
                       auto=0.15)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bloch_kernel, dim=(height, width),
              inputs=[img, cam, tip, tip2, int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.015))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="quantum_computer",
    description="A qubit on the Bloch sphere — a glowing wireframe sphere with X/Y/Z "
                "axes and a state vector precessing through superposition (|0⟩ pole "
                "to |1⟩ pole), plus an entangled partner. iMouse orbits; --frames animates.",
    renderer=_render,
)
