"""Aurora — volumetric light curtains over a night landscape.

A dark heightfield horizon (silhouette) under a starfield, with raymarched aurora
curtains in a high-altitude band: thin ridged filaments warped by flowing fbm,
a green-to-magenta vertical gradient, and ray streaks. Emissive (thin medium),
accumulated only through the aurora band. `--quality` scales the curtain samples.
iMouse pans / raises the curtains.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..lod import active_tier
from ..procedural.noise import fbm3, ridged3
from ..scene import Scene

_AH0 = 30.0
_AH1 = 95.0
_FAR = 260.0


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.02, 0.0, z * 0.02)
    return ridged3(p * 0.7, 5) * 12.0 + fbm3(p, 4) * 3.0 - 5.0


@wp.func
def _aurora(p: wp.vec3, time: float) -> wp.vec3:
    hf = wp.clamp((p[1] - _AH0) / (_AH1 - _AH0), 0.0, 1.0)
    warp = fbm3(wp.vec3(p[0] * 0.02, p[2] * 0.02, time * 0.15), 3)
    fil = ridged3(wp.vec3(p[0] * 0.03 + warp * 2.5, hf * 0.6, p[2] * 0.03), 4)
    curtain = wp.smoothstep(0.63, 0.92, fil)
    rays = wp.pow(0.5 + 0.5 * wp.sin(p[0] * 0.5 + p[2] * 0.3 + hf * 9.0), 1.5)
    vprof = wp.exp(-hf * 1.7) * rays
    green = wp.vec3(0.15, 1.0, 0.45)
    magenta = wp.vec3(0.7, 0.25, 1.0)
    col = green * (1.0 - hf) + magenta * hf
    return col * (curtain * vprof * 0.55)


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, time: float,
                  ground_steps: int, aur_steps: int, width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # ground silhouette
    t = float(1.0)
    hit = int(0)
    for _ in range(ground_steps):
        p = ro + rd * t
        if p[1] - _height(p[0], p[2]) < 0.0:
            hit = 1
            break
        t += wp.max((p[1] - _height(p[0], p[2])) * 0.5, 0.05 * t)
        if t > _FAR:
            break

    if hit == 1:
        col = wp.vec3(0.02, 0.03, 0.06)     # dark ground
    else:
        col = stars(rd)
        # accumulate aurora through the altitude band (upward rays only)
        if rd[1] > 0.001:
            te = (_AH0 - ro[1]) / rd[1]
            tx = (_AH1 - ro[1]) / rd[1]
            te = wp.max(te, 0.0)
            if tx > te:
                seg = (tx - te) / float(aur_steps)
                tt = te + 0.5 * seg
                acc = wp.vec3(0.0, 0.0, 0.0)
                for _ in range(aur_steps):
                    acc += _aurora(ro + rd * tt, time) * seg
                    tt += seg
                col = col + acc * 0.035
    img[i, j] = col


def _counts(name):
    return {"low": (90, 20), "medium": (140, 32), "high": (220, 48),
            "ultra": (320, 72)}.get(name, (140, 32))


def _render(width, height, time, mouse, device):
    tier = active_tier()
    gs, as_ = _counts(tier.name)

    az = 0.5 + float(mouse[0]) * 0.008
    pitch = 0.22 + float(mouse[1]) * 0.004
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    eye = (0.0, 6.0, 0.0)
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])
    cam = make_camera(eye, target, fov_deg=70.0, aspect=width / height)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(height, width),
              inputs=[img, cam, float(time), int(gs), int(as_), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.85, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.0)


SCENE = Scene(
    name="aurora",
    description="Aurora curtains over a night landscape + stars (volumetric). --quality low..ultra.",
    renderer=_render,
)
