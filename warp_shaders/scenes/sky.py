"""Sky — physically based atmospheric scattering from the ground.

Renders the Nishita/O'Neil single-scatter atmosphere with an animated sun that
sweeps below and above the horizon, so a time sequence runs sunrise → noon →
sunset → night with correct blue-zenith / orange-horizon shifts. Sample counts
scale with `--quality`. iMouse: x = look azimuth, y = raise/lower the sun.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..engine.atmosphere import (
    build_multiscatter_lut, build_transmittance_lut, sample_counts,
    sky_radiance_lut,
)
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..scene import Scene

_GROUND_EYE_Y = 6360002.0  # planet radius + 2 m
_lut_cache = {}


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  view_samples: int, lut: wp.array2d(dtype=wp.vec3),
                  ms_lut: wp.array2d(dtype=wp.vec3), width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = camera_ray_dir(cam, u, v)
    img[i, j] = sky_radiance_lut(cam.eye, rd, sun, view_samples, lut, ms_lut)


def _luts(device, size):
    """Transmittance + multiple-scattering LUTs (baked once, cached; same device)."""
    key = (device, size)
    if key not in _lut_cache:
        tr = build_transmittance_lut(size=size, device=device)
        ms = build_multiscatter_lut(tr, size=size, device=device)
        _lut_cache[key] = (tr, ms)
    return _lut_cache[key]


def _render(width, height, time, mouse, device):
    tier = active_tier()
    vs, ls = sample_counts(tier.name)
    lut, ms_lut = _luts(device, tier.lut_size)

    az = 0.3 + float(mouse[0]) * 0.01 + time * 0.03
    pitch = 0.22
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    eye = (0.0, _GROUND_EYE_Y, 0.0)
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=72.0, aspect=width / height)

    # sun sweeps from just above the horizon to near the zenith over a cycle
    el = 1.2 * math.sin(time * 0.12) + float(mouse[1]) * 0.004
    sun_np = np.array([math.sin(az) * math.cos(el), math.sin(el),
                       math.cos(az) * math.cos(el)], np.float32)
    sun = wp.vec3(float(sun_np[0]), float(sun_np[1]), float(sun_np[2]))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(vs), lut, ms_lut, int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=3.0, strength=0.5, radius=r, passes=3)

    # project the sun to screen space and add godray light shafts
    fwd_n = np.array(fwd, np.float32)
    fwd_n /= np.linalg.norm(fwd_n) + 1e-9
    right = np.cross(fwd_n, np.array([0, 1, 0], np.float32))
    right /= np.linalg.norm(right) + 1e-9
    upv = np.cross(right, fwd_n)
    cz = float(sun_np @ fwd_n)
    if cz > 0.02:
        thf = math.tan(math.radians(72.0) * 0.5)
        asp = width / height
        cx = 0.5 + 0.5 * (float(sun_np @ right) / cz) / (asp * thf)
        cy = 0.5 - 0.5 * (float(sun_np @ upv) / cz) / thf
        hdr = post.godrays(hdr, cx, cy, samples=32, threshold=2.5, weight=0.5)

    return post.tonemap(hdr, mode="aces", exposure=1.0)


SCENE = Scene(
    name="sky",
    description="Physically based atmospheric scattering (Rayleigh+Mie); animated sun. --quality low..ultra.",
    renderer=_render,
)
