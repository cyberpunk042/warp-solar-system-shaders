"""Twin suns setting over an alien sea — a binary-star seascape.

A ground-level view across an ocean at dusk: two suns (a big amber primary and
a smaller red companion) hang just over the horizon, laying a shivering double
**glitter path** across a wave-rippled sea. The water is an analytic plane whose
normals are perturbed by flowing fbm waves; it reflects the sky by Fresnel, so
the suns' reflections scatter into the classic sun-glint track. A warm-to-violet
dusk gradient, horizon haze. --frames rolls the swell.

Composes the engine's sky + a Fresnel water plane — a new subject, no globe, no
existing scene touched (distinct from ``twin_suns``, the living-meadow binary).
"""

import math

import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.func
def _sky(rd: wp.vec3, s0: wp.vec3, s1: wp.vec3) -> wp.vec3:
    up = wp.clamp(rd[1] * 2.6, 0.0, 1.0)
    horizon = wp.vec3(0.95, 0.38, 0.18)
    zenith = wp.vec3(0.05, 0.05, 0.15)
    base = horizon * (1.0 - up) + zenith * up
    # a brighter band hugging the horizon (forward-scattered dusk light)
    band = wp.exp(-wp.abs(rd[1]) * 16.0)
    base = base + wp.vec3(1.0, 0.5, 0.22) * (band * 0.25)
    # the two suns: sharp disks + tight aureoles
    d0 = wp.max(wp.dot(rd, s0), 0.0)
    d1 = wp.max(wp.dot(rd, s1), 0.0)
    base = base + wp.vec3(1.0, 0.78, 0.45) * (wp.pow(d0, 2600.0) * 70.0 + wp.pow(d0, 26.0) * 0.35)
    base = base + wp.vec3(1.0, 0.40, 0.26) * (wp.pow(d1, 4200.0) * 45.0 + wp.pow(d1, 30.0) * 0.28)
    return base


@wp.func
def _wave_h(x: float, z: float, time: float) -> float:
    a = fbm3(wp.vec3(x * 0.15 + time * 0.10, 0.0, z * 0.15), 4) * 0.16
    b = fbm3(wp.vec3(x * 0.55, time * 0.22, z * 0.55), 3) * 0.05
    return a + b


@wp.func
def _wave_n(p: wp.vec3, time: float) -> wp.vec3:
    e = 0.18
    hx = _wave_h(p[0] + e, p[2], time) - _wave_h(p[0] - e, p[2], time)
    hz = _wave_h(p[0], p[2] + e, time) - _wave_h(p[0], p[2] - e, time)
    return wp.normalize(wp.vec3(-hx, 2.0 * e, -hz))


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, s0: wp.vec3,
                  s1: wp.vec3, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    if rd[1] < -0.004:                                  # ray meets the sea plane y=0
        t = -ro[1] / rd[1]
        p = ro + rd * t
        # wave normal (flatten with distance so far water stays glassy, not noisy)
        near = wp.exp(-t * 0.03)
        n = _wave_n(p, time)
        n = wp.normalize(wp.vec3(n[0] * near, 1.0, n[2] * near))
        r = rd - n * (2.0 * wp.dot(rd, n))
        r = wp.vec3(r[0], wp.abs(r[1]), r[2])           # keep reflection in the sky
        refl = _sky(r, s0, s1)
        deep = wp.vec3(0.015, 0.045, 0.085)
        f = 1.0 - wp.max(wp.dot(-rd, n), 0.0)
        fres = 0.02 + 0.98 * wp.pow(f, 5.0)             # Schlick
        col = deep * (1.0 - fres) + refl * fres
        fog = 1.0 - wp.exp(-t * 0.02)                   # fade to sky at the horizon
        col = col * (1.0 - fog) + _sky(rd, s0, s1) * fog
    else:
        col = _sky(rd, s0, s1)

    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.0 + float(mouse[0]) * 0.006
    pitch = 0.03 + float(mouse[1]) * 0.003
    eye = (0.0, 3.2, 0.0)
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch) + 0.06, math.cos(az) * math.cos(pitch))
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])

    # two suns just over the horizon, close together
    a0 = az - 0.10
    a1 = az + 0.14
    e0 = 0.045
    e1 = 0.085
    s0 = wp.vec3(math.sin(a0) * math.cos(e0), math.sin(e0), math.cos(a0) * math.cos(e0))
    s1 = wp.vec3(math.sin(a1) * math.cos(e1), math.sin(e1), math.cos(a1) * math.cos(e1))

    ss = 2
    W, H = int(width) * ss, int(height) * ss
    cam = make_camera(eye, target, fov_deg=62.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, s0, s1, float(time), int(W), int(H)], device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss)
    r = max(3, int(min(width, height) * 0.018))
    hdr = post.bloom(hdr, threshold=1.3, strength=0.5, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=0.82, preserve_hue=True)


SCENE = Scene(
    name="binary_sea",
    description="Twin suns setting over an alien sea — a big amber primary and a "
                "small red companion low on the horizon, laying a double glitter "
                "path across a wave-rippled Fresnel ocean at dusk. --frames rolls "
                "the swell.",
    renderer=_render,
)
