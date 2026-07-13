"""The cosmic microwave background — the afterglow of the Big Bang.

The whole sky at 380,000 years, redshifted to 2.725 K, shown as a sphere painted
with its tiny **temperature anisotropies** (ΔT/T ~ 10⁻⁵) — an fBm fluctuation
field false-coloured in the iconic Planck palette (cold blue → hot red), with a
faint dipole. These specks are the seeds that grew into galaxies. See
``docs/research/23-origin-and-large-scale-universe.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.intersect import ray_sphere_o as _rs
from ..engine.uniforms import Camera, camera_ray_dir
from ..procedural.noise import fbm3
from ..subatomic.render import orbit_camera
from ..scene import Scene


@wp.func
def _planck(t: float) -> wp.vec3:
    # cold blue → cyan → green → yellow → red (WMAP/Planck false colour)
    t = wp.clamp(t, 0.0, 1.0)
    if t < 0.25:
        k = t / 0.25
        return wp.vec3(0.05, 0.1, 0.6) * (1.0 - k) + wp.vec3(0.1, 0.7, 0.9) * k
    if t < 0.5:
        k = (t - 0.25) / 0.25
        return wp.vec3(0.1, 0.7, 0.9) * (1.0 - k) + wp.vec3(0.2, 0.8, 0.3) * k
    if t < 0.75:
        k = (t - 0.5) / 0.25
        return wp.vec3(0.2, 0.8, 0.3) * (1.0 - k) + wp.vec3(0.95, 0.85, 0.2) * k
    k = (t - 0.75) / 0.25
    return wp.vec3(0.95, 0.85, 0.2) * (1.0 - k) + wp.vec3(0.9, 0.15, 0.1) * k


@wp.kernel
def cmb_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
               width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    g = _rs(ro, rd, 1.0)
    if g[0] > 1.0e28 or g[0] < 0.0:
        img[i, j] = wp.vec3(0.01, 0.011, 0.02)
        return
    n = wp.normalize(ro + rd * g[0])
    # multi-scale temperature anisotropies
    fine = fbm3(n * 7.0, 6)
    coarse = fbm3(n * 2.3 + wp.vec3(3.1, 1.7, 5.2), 4)
    aniso = 0.6 * fine + 0.4 * coarse
    dipole = 0.05 * n[0]                               # the CMB dipole (our motion)
    t = wp.clamp(0.5 + (aniso - 0.5) * 2.6 + dipole, 0.0, 1.0)   # widen to blue↔red
    col = _planck(t)
    # limb darkening so it reads as a sphere
    limb = wp.pow(wp.max(wp.dot(n, -rd), 0.0), 0.35)
    img[i, j] = col * (0.35 + 0.75 * limb)


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=2.55, fov=46.0, el0=0.15,
                       auto=0.06)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(cmb_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=1.6, strength=0.15, radius=2, passes=1)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="cmb",
    description="The cosmic microwave background — the 380,000-year-old sky as a "
                "sphere painted with its temperature anisotropies (ΔT/T~1e-5) in "
                "the Planck false-colour palette. The seeds of all structure.",
    renderer=_render,
)
