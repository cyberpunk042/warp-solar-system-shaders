"""Ferrofluid — a liquid that spikes along magnetic field lines.

A ferrofluid is a colloid of magnetic nanoparticles. Put it in a vertical magnetic
field and its surface erupts into a self-organising hexagonal field of **spikes** —
the **Rosensweig instability** — each peak riding a field line, the black fluid glossy
with sharp specular highlights. See ``docs/research/31-states-of-matter.md``.
--frames ramps the field up and down.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


@wp.func
def _spikes(x: float, z: float, amp: float) -> float:
    h = float(0.0)
    sx = 0.5
    sz = 0.44
    cj0 = wp.floor(z / sz + 0.5)
    for dj in range(-1, 2):
        cj = cj0 + float(dj)
        off = 0.5 * wp.mod(cj, 2.0) * sx
        ci0 = wp.floor((x - off) / sx + 0.5)
        for di in range(-1, 2):
            ci = ci0 + float(di)
            cx = ci * sx + off
            cz = cj * sz
            d2 = (x - cx) * (x - cx) + (z - cz) * (z - cz)
            h = h + wp.exp(-d2 / 0.024) * amp
    return h


@wp.kernel
def ferro_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                 rgt: wp.vec3, upv: wp.vec3, aspect: float, amp: float,
                 width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = wp.normalize(fwd * 1.5 + rgt * (u * aspect) + upv * v)

    # sky/studio gradient (reflected by the glossy fluid)
    sky = wp.vec3(0.08, 0.10, 0.16) + wp.vec3(0.5, 0.55, 0.65) * wp.pow(wp.max(rd[1], 0.0), 2.0)
    col = sky

    t = float(0.1)
    hit = int(0)
    hp = wp.vec3(0.0, 0.0, 0.0)
    for _ in range(150):
        p = eye + rd * t
        h = _spikes(p[0], p[2], amp)
        if p[1] < h and p[0] > -2.3 and p[0] < 2.3 and p[2] > -2.3 and p[2] < 2.3:
            hit = 1
            hp = p
            break
        t += wp.max((p[1] - h) * 0.35, 0.015)
        if t > 12.0:
            break

    if hit == 1:
        e = 0.01
        n = wp.normalize(wp.vec3(
            _spikes(hp[0] - e, hp[2], amp) - _spikes(hp[0] + e, hp[2], amp),
            2.0 * e,
            _spikes(hp[0], hp[2] - e, amp) - _spikes(hp[0], hp[2] + e, amp)))
        L = wp.normalize(wp.vec3(0.35, 0.85, 0.35))
        hlf = wp.normalize(L - rd)
        spec = wp.pow(wp.max(wp.dot(n, hlf), 0.0), 60.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 4.0)
        refl = rd - n * (2.0 * wp.dot(rd, n))
        envr = wp.vec3(0.1, 0.12, 0.18) + wp.vec3(0.5, 0.55, 0.65) * wp.pow(wp.max(refl[1], 0.0), 2.0)
        col = wp.vec3(0.015, 0.016, 0.022) + envr * (0.35 + 0.5 * fres) \
            + wp.vec3(1.0, 1.0, 1.0) * spec * 1.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    amp = 0.22 + 0.42 * (0.5 + 0.5 * float(np.sin(time * 0.6)))
    eye = wp.vec3(0.0, 1.7, 2.7)
    tgt = wp.vec3(0.0, 0.25, 0.0)
    fwd = wp.normalize(tgt - eye)
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(ferro_kernel, dim=(height, width),
              inputs=[img, eye, fwd, rgt, upv, float(width / height), float(amp),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="ferrofluid",
    description="A ferrofluid spiking in a magnetic field — the Rosensweig instability: "
                "a black magnetic liquid erupting into a self-organising hexagonal field "
                "of glossy spikes riding the field lines. --frames ramps the field.",
    renderer=_render,
)
