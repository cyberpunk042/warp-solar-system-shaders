"""A great comet over a dark mountain ridge — ion and dust tails in the night sky.

A ground-level night scene: a bright comet hangs over a silhouetted mountain
horizon under a dense starfield. The comet shows the two tails a real comet grows
near the Sun — a straight, narrow **ion tail** (blue CO+ fluorescence, blown
dead anti-sunward by the solar wind) and a broader, **curved dust tail** (yellow-
white, lagging along the orbit) — streaming from a green-tinged **coma** around a
brilliant nucleus. --frames drifts the comet across the sky.

Composes the heightfield silhouette + starfield with a procedural comet — a new
subject, no existing scene touched.
"""

import math

import warp as wp

from ..earthgfx import stars
from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir, make_camera
from ..procedural.noise import fbm3, ridged3
from ..scene import Scene

_FAR = 300.0


@wp.func
def _height(x: float, z: float) -> float:
    p = wp.vec3(x * 0.02, 0.0, z * 0.02)
    return ridged3(p * 0.6, 5) * 10.0 + fbm3(p, 4) * 2.0 - 4.0


@wp.func
def _comet(rd: wp.vec3, cdir: wp.vec3, right: wp.vec3, upv: wp.vec3) -> wp.vec3:
    f = wp.dot(rd, cdir)
    if f < 0.2:
        return wp.vec3(0.0, 0.0, 0.0)
    x = wp.dot(rd, right)
    y = wp.dot(rd, upv)
    r2 = x * x + y * y

    # coma + nucleus (the tails stream in the +y / "up-and-away" direction)
    core = wp.exp(-r2 / 0.00035) * 4.0
    coma = wp.exp(-r2 / 0.006) * 0.7
    col = wp.vec3(0.85, 1.0, 0.9) * core + wp.vec3(0.4, 0.9, 0.7) * coma

    ty = wp.max(y, 0.0)
    m = wp.smoothstep(-0.005, 0.03, y)              # tails only stream upward
    # straight narrow blue ion tail, blown dead anti-sunward
    ion = wp.exp(-(x * x) / 0.0010) * wp.exp(-ty * 2.6) * m
    col = col + wp.vec3(0.45, 0.62, 1.0) * (ion * 1.4)
    # broader curved yellow-white dust tail, lagging along the orbit
    cx = x - 0.7 * ty * ty
    dust = wp.exp(-(cx * cx) / 0.006) * wp.exp(-ty * 2.0) * m
    col = col + wp.vec3(1.0, 0.9, 0.7) * (dust * 0.9)
    return col


@wp.kernel
def render_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera, cdir: wp.vec3,
                  right: wp.vec3, upv: wp.vec3, march_steps: int,
                  width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)

    # dark mountain silhouette
    t = float(1.0)
    hit = int(0)
    for _ in range(march_steps):
        p = ro + rd * t
        d = p[1] - _height(p[0], p[2])
        if d < 0.0:
            hit = 1
            break
        t += wp.max(d * 0.4, 0.01 * t)
        if t > _FAR:
            break

    if hit == 1:
        col = wp.vec3(0.012, 0.014, 0.028)              # near-black ridge
    else:
        # night sky: deep blue gradient + stars + the comet
        up = wp.clamp(rd[1] * 0.8 + 0.2, 0.0, 1.0)
        col = wp.vec3(0.03, 0.04, 0.08) * (1.0 - up) + wp.vec3(0.01, 0.015, 0.05) * up
        col = col + stars(rd)
        col = col + _comet(rd, cdir, right, upv)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    az = 0.3 + float(mouse[0]) * 0.006 + time * 0.02
    pitch = 0.28 + float(mouse[1]) * 0.003
    eye = (0.0, 14.0, 0.0)
    fwd = (math.sin(az) * math.cos(pitch), math.sin(pitch), math.cos(az) * math.cos(pitch))
    target = (eye[0] + fwd[0], eye[1] + fwd[1], eye[2] + fwd[2])

    # comet direction: ahead and up; build a screen basis around it, with the
    # tails streaming toward the top of the sky (away from the sun below horizon)
    cel = 0.42
    cdir = wp.vec3(math.sin(az + 0.15) * math.cos(cel), math.sin(cel),
                   math.cos(az + 0.15) * math.cos(cel))
    upv = wp.vec3(0.0, 1.0, 0.0)
    right = wp.normalize(wp.cross(upv, cdir))
    upv = wp.normalize(wp.cross(cdir, right))

    ss = 2
    W, H = int(width) * ss, int(height) * ss
    cam = make_camera(eye, target, fov_deg=64.0, aspect=W / H)
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(render_kernel, dim=(H, W),
              inputs=[img, cam, cdir, right, upv, int(220), int(W), int(H)], device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy(), ss)
    r = max(3, int(min(width, height) * 0.016))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.5, radius=r, passes=3, octaves=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="comet",
    description="A great comet over a dark mountain ridge — a brilliant nucleus in "
                "a green coma trailing a straight blue ion tail and a curved yellow "
                "dust tail across a starfield. --frames drifts it across the sky.",
    renderer=_render,
)
