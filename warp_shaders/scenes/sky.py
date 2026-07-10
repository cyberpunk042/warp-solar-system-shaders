"""Sky — physically based atmospheric scattering from the ground.

Renders the Nishita/O'Neil single-scatter atmosphere with an animated sun that
sweeps below and above the horizon, so a time sequence runs sunrise → noon →
sunset → night with correct blue-zenith / orange-horizon shifts. Sample counts
scale with `--quality`. iMouse: x = look azimuth, y = raise/lower the sun.
"""

import math

import warp as wp

from ..engine import post
from ..engine.atmosphere import build_transmittance_lut, sample_counts, sky_radiance_lut
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..scene import Scene

_GROUND_EYE_Y = 6360002.0  # planet radius + 2 m
_lut_cache = {}


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, sun: wp.vec3,
                  view_samples: int, lut: wp.array2d(dtype=wp.vec3),
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = camera_ray_dir(cam, u, v)
    img[i, j] = sky_radiance_lut(cam.eye, rd, sun, view_samples, lut)


def _lut(device, size):
    key = (device, size)
    if key not in _lut_cache:
        _lut_cache[key] = build_transmittance_lut(size=size, device=device)
    return _lut_cache[key]


def _render(width, height, time, mouse, device):
    tier = active_tier()
    vs, ls = sample_counts(tier.name)
    lut = _lut(device, tier.lut_size)

    az = 0.3 + float(mouse[0]) * 0.01 + time * 0.03
    pitch = 0.22
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    eye = (0.0, _GROUND_EYE_Y, 0.0)
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=72.0, aspect=width / height)

    # sun sweeps from just above the horizon to near the zenith over a cycle
    el = 1.2 * math.sin(time * 0.12) + float(mouse[1]) * 0.004
    sun = wp.vec3(math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, sun, int(vs), lut, int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()

    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=3.0, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.0)


SCENE = Scene(
    name="sky",
    description="Physically based atmospheric scattering (Rayleigh+Mie); animated sun. --quality low..ultra.",
    renderer=_render,
)
