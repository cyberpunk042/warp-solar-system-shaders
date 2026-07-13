"""An O'Neill cylinder — a world spun inside a bottle.

Gerard O'Neill's rotating habitat (1976): a cylinder kilometres across whose **spin**
presses you to the inner wall at 1 g. The land wraps up and over your head — a valley
curving into the sky — in alternating **land strips** and **window strips** that let
sunlight in, lit by a glowing sun-tube along the axis. See
``docs/research/29-megastructures-and-far-future.md``. --frames drifts the light.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..scene import Scene

_R = 3.2


@wp.kernel
def cyl_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
               rgt: wp.vec3, upv: wp.vec3, aspect: float, time: float,
               width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = wp.normalize(fwd * 1.5 + rgt * (u * aspect) + upv * v)

    col = wp.vec3(0.02, 0.03, 0.05)
    # ray vs cylinder inner wall (axis = z), outgoing root
    a = rd[0] * rd[0] + rd[1] * rd[1]
    b = 2.0 * (eye[0] * rd[0] + eye[1] * rd[1])
    c = eye[0] * eye[0] + eye[1] * eye[1] - _R * _R
    disc = b * b - 4.0 * a * c
    twall = float(1e30)
    if disc > 0.0 and a > 1e-6:
        twall = (-b + wp.sqrt(disc)) / (2.0 * a)
        if twall > 0.0:
            p = eye + rd * twall
            ang = wp.atan2(p[1], p[0])
            sector = wp.floor((ang / 6.2831 + 0.5) * 6.0)
            zz = p[2]
            haze = 1.0 - wp.exp(-wp.abs(zz - eye[2]) * 0.02)
            if wp.mod(sector, 2.0) < 0.5:                 # land strip
                terr = fbm3(wp.vec3(ang * 3.0, zz * 0.3, 0.0), 4)
                rivers = wp.smoothstep(0.48, 0.5, fbm3(wp.vec3(ang * 5.0, zz * 0.5, 4.0), 3))
                green = wp.vec3(0.16, 0.46, 0.14) * (0.6 + 0.9 * terr)
                brown = wp.vec3(0.35, 0.28, 0.16)
                land = green * (1.0 - 0.4 * terr) + brown * (0.4 * terr)
                water = wp.vec3(0.1, 0.25, 0.4)
                surf = land * rivers + water * (1.0 - rivers)
                col = surf
            else:                                         # window strip (sunlight in)
                stripe = 0.6 + 0.4 * wp.sin((ang + 0.5) * 40.0)
                col = wp.vec3(0.4, 0.58, 0.85) * (0.7 + 0.4 * stripe)
            col = col * (1.0 - 0.45 * haze) + wp.vec3(0.35, 0.5, 0.72) * 0.45 * haze

    # glowing sun-tube along the axis (x=y=0 line)
    rxy2 = rd[0] * rd[0] + rd[1] * rd[1]
    if rxy2 > 1e-6:
        ts = -(eye[0] * rd[0] + eye[1] * rd[1]) / rxy2
        if ts > 0.0 and ts < twall:
            cxy = wp.length(wp.vec2(eye[0] + rd[0] * ts, eye[1] + rd[1] * ts))
            glow = wp.exp(-(cxy / 0.22) * (cxy / 0.22))
            col = col + wp.vec3(1.0, 0.95, 0.75) * glow * 1.8

    img[i, j] = col


def _render(width, height, time, mouse, device):
    ang = time * 0.05
    eye = wp.vec3(0.0, -2.5, -7.0)
    tgt = wp.vec3(0.35 * float(np.sin(ang)), 0.4, 3.5)
    fwd = wp.normalize(tgt - eye)
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(cyl_kernel, dim=(height, width),
              inputs=[img, eye, fwd, rgt, upv, float(width / height), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="oneill_cylinder",
    description="Inside an O'Neill cylinder — a rotating habitat where the land wraps up "
                "and over your head, alternating green land strips and sunlit window "
                "strips lit by a glowing sun-tube along the axis, the valley curving into "
                "the sky. --frames drifts the light.",
    renderer=_render,
)
