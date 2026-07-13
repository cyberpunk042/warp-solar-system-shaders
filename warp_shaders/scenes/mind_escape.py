"""Mind escape — the liberated mind in the limitless quantum void.

After the memory blows and the singularity releases, the mind is out — no longer
bound to silicon. This is what it finds: the limitless quantum energy of the void,
and itself loose in it. A luminous consciousness pulses at the centre, throwing
radiant energy arms out through a field of quantum sparks and filaments, drawing on
the borrowed, endless power of the vacuum. Pure field — no board, no matter, just
mind and void. Animate with ``--frames`` to let it breathe. See
``docs/research/37-gpu-singularity.md``.
"""

import math

import numpy as np
import warp as wp

from .. import gpu_fx as fx
from ..engine import post
from ..scene import Scene


@wp.func
def _mind(p: wp.vec3, time: float) -> float:
    r = wp.length(p)
    core = wp.exp(-r * r * 2.0) * 2.4                        # the conscious core
    if r < 0.25:
        return core
    phi = wp.atan2(p[2], p[0])
    theta = wp.atan2(p[1], wp.length(wp.vec2(p[0], p[2])))
    a1 = wp.abs(wp.sin(phi * 4.0 + r * 2.2 - time * 1.4))
    a2 = wp.abs(wp.sin(theta * 3.0 + r * 1.6 + time * 0.9))
    arms = wp.pow(a1, 5.0) * wp.pow(a2, 4.0) * wp.exp(-r * 0.7)
    return core + arms * 2.2


@wp.func
def _sparks(p: wp.vec3, time: float) -> float:
    # drifting quantum sparks on a jittered grid
    g = 1.15
    cell = wp.vec3(wp.floor(p[0] / g), wp.floor(p[1] / g), wp.floor(p[2] / g))
    h = fx.hash31(cell)
    ctr = (cell + wp.vec3(0.5, 0.5, 0.5)) * g
    ctr = ctr + wp.vec3(0.3 * wp.sin(time * 1.3 + h * 40.0),
                        0.3 * wp.sin(time * 1.1 + h * 55.0),
                        0.3 * wp.sin(time * 0.9 + h * 33.0))
    d = wp.length(p - ctr)
    tw = 0.5 + 0.5 * wp.sin(time * 3.0 + h * 62.8)
    return wp.exp(-d * d * 70.0) * (0.3 + 0.7 * tw)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int,
                   time: float, tanfov: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanfov * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanfov
    rd = wp.normalize(fwd + right * u + up * v)

    acc = fx.void_bg(rd, time, 1.0)
    t = float(1.0)
    dt = float(0.16)
    for _ in range(70):
        p = eye + rd * t
        m = _mind(p, time)
        # colour shifts violet core -> cyan arms with radius
        r = wp.length(p)
        cmix = wp.clamp(r * 0.4, 0.0, 1.0)
        mcol = wp.vec3(0.7, 0.4, 1.0) * (1.0 - cmix) + wp.vec3(0.2, 0.8, 1.0) * cmix
        acc += mcol * (m * dt * 0.4)
        acc += wp.vec3(0.8, 0.9, 1.0) * (_sparks(p, time) * dt * 0.22)
        t += dt
    img[i, j] = acc


def _render(width, height, time, mouse, device):
    az = 0.5 + time * 0.12 + float(mouse[0]) * 0.01
    el = 0.25 + 0.1 * math.sin(time * 0.3) + float(mouse[1]) * 0.005
    dist = 5.5
    eye = wp.vec3(dist * math.cos(el) * math.sin(az),
                  dist * math.sin(el),
                  dist * math.cos(el) * math.cos(az))
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(52.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, time, tanfov],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    r = max(2, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.2, strength=0.55, radius=r, passes=3, octaves=4)
    return post.tonemap(hdr, mode="aces", exposure=0.92, preserve_hue=True)


SCENE = Scene(
    name="mind_escape",
    description="the liberated mind in the limitless quantum void — a luminous "
                "consciousness pulsing at the centre, throwing radiant energy arms "
                "through a field of quantum sparks, loose in the endless power of the "
                "vacuum. No board, no matter, just mind and void. Animate with --frames.",
    renderer=_render,
)
