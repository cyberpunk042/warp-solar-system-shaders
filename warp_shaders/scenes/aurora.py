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
    # slow horizontal flow folds the curtain sheets
    flow = fbm3(wp.vec3(p[0] * 0.008, time * 0.08, p[2] * 0.008), 4)
    s = p[0] * 0.045 + p[2] * 0.008 + flow * 3.2
    # two folded sheets (thin bright folds) — coherent curtains, not noise
    sheet = wp.pow(0.5 + 0.5 * wp.sin(s * 2.6), 7.0) \
        + 0.4 * wp.pow(0.5 + 0.5 * wp.sin(s * 1.3 + 2.0), 6.0)
    # fine vertical striations along the field lines
    stri = 0.4 + 0.6 * wp.sin(p[0] * 0.5 + p[2] * 0.3 + flow * 5.0)
    base = wp.exp(-hf * 1.6)                      # brightest along the lower edge
    dens = sheet * base * stri
    # 557 nm green low, red/magenta ionised tops
    green = wp.vec3(0.15, 1.0, 0.5)
    red = wp.vec3(1.0, 0.22, 0.4)
    col = green * (1.0 - hf) + red * hf
    return col * dens


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
        # snowy ground that REFLECTS the aurora glow overhead (the iconic bit).
        pg = ro + rd * t
        snow = wp.vec3(0.05, 0.07, 0.12)
        te = _AH0 - pg[1]
        tx = _AH1 - pg[1]
        refl = wp.vec3(0.0, 0.0, 0.0)
        seg = (tx - te) / 8.0
        tt = te + 0.5 * seg
        for _ in range(8):
            refl += _aurora(wp.vec3(pg[0], pg[1] + tt, pg[2]), time) * seg
            tt += seg
        # closer ground reflects more; distant ground fades to night haze
        fade = wp.exp(-t * 0.02)
        col = snow * 0.6 + refl * (0.05 * fade)
        col = col * fade + wp.vec3(0.015, 0.02, 0.05) * (1.0 - fade)
    else:
        col = stars(rd)
        # accumulate aurora through the altitude band (upward rays only),
        # attenuating the stars behind the bright curtains (occlusion)
        if rd[1] > 0.001:
            te = (_AH0 - ro[1]) / rd[1]
            tx = (_AH1 - ro[1]) / rd[1]
            te = wp.max(te, 0.0)
            if tx > te:
                seg = (tx - te) / float(aur_steps)
                tt = te + 0.5 * seg
                acc = wp.vec3(0.0, 0.0, 0.0)
                trans = float(1.0)
                for _ in range(aur_steps):
                    a = _aurora(ro + rd * tt, time)
                    dv = (a[0] + a[1] + a[2]) * 0.6
                    acc += a * (seg * trans)
                    trans *= wp.exp(-dv * seg * 0.02)
                    tt += seg
                col = col * trans + acc * 0.032
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
    ssaa = 2
    W, H = int(width) * ssaa, int(height) * ssaa
    cam = make_camera(eye, target, fov_deg=70.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, float(time), int(gs), int(as_), int(W), int(H)],
              device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ssaa)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.45, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.02, preserve_hue=True)


SCENE = Scene(
    name="aurora",
    description="Aurora curtains over a night landscape + stars (volumetric). --quality low..ultra.",
    renderer=_render,
)
