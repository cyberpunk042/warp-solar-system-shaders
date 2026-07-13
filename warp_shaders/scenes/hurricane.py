"""A hurricane — the satellite view of a tropical cyclone.

Log-spiral **rainbands** wheeling counter-clockwise around a clear, calm **eye**,
ringed by the bright towering **eyewall**, over warm ocean. The cloud tops are
shaded in 3D (a normal lifted from the cloud-density gradient, lit by a low sun),
so the spiral bands cast their own light and shadow and read as towering convection
rather than a flat white swirl; deep tropical ocean shows in the gaps and the calm
eye. Spins with `time`. See ``docs/research/25-earth-and-weather.md``.
--frames animates the rotation.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..scene import Scene

_EYE = wp.constant(0.12)


@wp.func
def _cloud(x: float, y: float, rot: float, time: float) -> float:
    r = wp.sqrt(x * x + y * y)
    th = wp.atan2(y, x)
    # tightening log-spiral rainbands
    spiral = th * 2.0 + wp.log(r + 0.05) * 7.5 - rot
    band = wp.pow(0.5 + 0.5 * wp.sin(spiral), 1.6)
    detail = fbm3(wp.vec3(x * 4.2, y * 4.2, time * 0.1), 5)
    # radial envelope: nothing in the eye, fades at the storm edge
    env = wp.smoothstep(_EYE * 0.7, _EYE * 1.25, r) * (1.0 - wp.smoothstep(0.78, 1.12, r))
    dens = wp.clamp((band * 0.75 + detail * 0.6 - 0.52) * 2.3, 0.0, 1.0)
    # central dense overcast: a solid cloud mass hugging the eyewall, thinning
    # outward into discrete spiral bands
    fill = 1.0 - wp.smoothstep(0.14, 0.46, r)
    dens = wp.max(dens, fill * (0.82 + 0.18 * detail))
    dens = dens * env
    # the towering eyewall ring around the eye
    ew = wp.exp(-((r - _EYE) / 0.03) * ((r - _EYE) / 0.03)) * env
    return wp.max(dens, ew * 0.95)


@wp.kernel
def hurricane_kernel(img: wp.array2d(dtype=wp.vec3), rot: float, aspect: float,
                     time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0
    r = wp.sqrt(x * x + y * y)

    d = _cloud(x, y, rot, time)
    # cloud-top normal from the density gradient → 3D shading
    e = 0.006
    gx = _cloud(x + e, y, rot, time) - _cloud(x - e, y, rot, time)
    gy = _cloud(x, y + e, rot, time) - _cloud(x, y - e, rot, time)
    n = wp.normalize(wp.vec3(-gx, -gy, 0.12))
    sun = wp.normalize(wp.vec3(0.55, 0.5, 0.5))
    diff = wp.max(wp.dot(n, sun), 0.0)
    shade = 0.55 + 0.7 * diff                       # ambient + directional
    cloudc = wp.vec3(0.95, 0.96, 1.0) * shade

    # warm tropical ocean, textured; brighter/calmer sunlit sea in the eye
    oc = 0.5 + 0.5 * fbm3(wp.vec3(x * 6.0, y * 6.0, time * 0.05), 4)
    ocean = wp.vec3(0.02, 0.11, 0.22) * (0.75 + 0.4 * oc)
    edge = 1.0 - wp.smoothstep(0.9, 1.25, r)
    eye_calm = (1.0 - wp.smoothstep(_EYE * 0.5, _EYE, r)) * edge
    ocean = ocean + wp.vec3(0.06, 0.12, 0.16) * eye_calm

    col = ocean * (1.0 - d) + cloudc * d
    # the eyewall's tallest towers catch a bright rim of sun
    ew = wp.exp(-((r - _EYE) / 0.028) * ((r - _EYE) / 0.028)) * edge
    col = col + wp.vec3(1.0, 1.0, 1.0) * (ew * diff * 0.4)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    rot = time * 1.3
    ss = 2
    W, H = int(width) * ss, int(height) * ss
    img = wp.zeros((H, W), dtype=wp.vec3, device=device)
    wp.launch(hurricane_kernel, dim=(H, W),
              inputs=[img, float(rot), float(W / H), float(time), int(W), int(H)],
              device=device)
    wp.synchronize_device(device)
    hdr = post.downsample(img.numpy().astype(np.float32), ss)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.4, strength=0.2, radius=r, passes=2, octaves=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05, preserve_hue=True)


SCENE = Scene(
    name="hurricane",
    description="A hurricane from orbit — log-spiral rainbands with 3D-shaded cloud "
                "tops wheeling around a clear eye ringed by a bright eyewall, over "
                "warm tropical ocean. --frames animates the rotation.",
    renderer=_render,
)
