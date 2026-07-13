"""A MOSFET — the switch everything is built from.

A cross-section of a **metal-oxide-semiconductor field-effect transistor**: a
**source** and **drain** doped into a silicon **substrate**, a **gate** electrode
on top separated by a thin **oxide**. Raise the gate voltage and an inversion
**channel** forms between source and drain — the switch turns **ON** and current
(pulses) flows; drop it and the channel vanishes (**OFF**). See
``docs/research/26-the-machine.md``. --frames toggles the gate.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..scene import Scene


@wp.func
def _box(p: wp.vec2, c: wp.vec2, b: wp.vec2) -> float:
    d = wp.vec2(wp.abs(p[0] - c[0]) - b[0], wp.abs(p[1] - c[1]) - b[1])
    return wp.length(wp.vec2(wp.max(d[0], 0.0), wp.max(d[1], 0.0))) + wp.min(wp.max(d[0], d[1]), 0.0)


@wp.func
def _edge(d: float, w: float) -> float:
    return wp.exp(-(d / w) * (d / w))


@wp.kernel
def fet_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, gate: float,
               time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)

    # dark silicon substrate with faint crystalline mottle
    sub = 0.5 + 0.5 * fbm3(wp.vec3(x * 4.0, y * 4.0, 0.0), 4)
    col = wp.vec3(0.03, 0.05, 0.07) + wp.vec3(0.02, 0.03, 0.03) * sub
    # p-type body region (lower slab)
    if y < 0.0:
        col = wp.vec3(0.06, 0.07, 0.11) + wp.vec3(0.02, 0.02, 0.03) * sub

    # source (left) & drain (right) n+ contacts
    ds = _box(p, wp.vec2(-0.62, -0.02), wp.vec2(0.2, 0.26))
    dd = _box(p, wp.vec2(0.62, -0.02), wp.vec2(0.2, 0.26))
    contact = wp.vec3(0.55, 0.7, 0.85)
    col = col + contact * (_edge(ds, 0.014) + _edge(dd, 0.014)) * 0.8
    if ds < 0.0:
        col = wp.vec3(0.1, 0.16, 0.22) + wp.vec3(0.08, 0.12, 0.16) * sub
    if dd < 0.0:
        col = wp.vec3(0.1, 0.16, 0.22) + wp.vec3(0.08, 0.12, 0.16) * sub

    # thin gate oxide + gate electrode (metal bar on top)
    dg = _box(p, wp.vec2(0.0, 0.42), wp.vec2(0.44, 0.12))
    gatecol = wp.vec3(0.6, 0.6, 0.7) + wp.vec3(0.6, 0.5, 0.2) * gate  # warms when driven
    if dg < 0.0:
        col = gatecol * (0.5 + 0.3 * sub)
    col = col + wp.vec3(0.9, 0.85, 0.7) * _edge(dg, 0.012) * (0.5 + gate)
    # oxide line
    dox = wp.abs(y - 0.28) + wp.max(wp.abs(x) - 0.44, 0.0) * 4.0
    col = col + wp.vec3(0.7, 0.5, 0.9) * _edge(dox, 0.01) * 0.5

    # the inversion channel: a bright strip under the gate, ON only when gate high
    chan = _box(p, wp.vec2(0.0, 0.18), wp.vec2(0.42, 0.05))
    on = wp.smoothstep(0.45, 0.75, gate)
    glow = _edge(chan, 0.06) * on
    col = col + wp.vec3(0.3, 0.95, 1.0) * glow * 1.6
    # current pulses drifting source -> drain along the channel when ON
    if wp.abs(y - 0.18) < 0.06:
        ph = wp.mod(x * 1.4 - time * 1.6, 0.5)
        pul = wp.pow(wp.max(1.0 - wp.abs(ph - 0.25) * 10.0, 0.0), 3.0)
        col = col + wp.vec3(0.8, 1.0, 1.0) * pul * on * 2.4

    # ON/OFF tell: a small indicator glow above the gate
    ind = wp.length(p - wp.vec2(0.0, 0.72)) - 0.03
    col = col + wp.vec3(0.4, 1.0, 0.5) * _edge(ind, 0.02) * on * 2.0
    col = col + wp.vec3(1.0, 0.3, 0.3) * _edge(ind, 0.02) * (1.0 - on) * 1.2

    img[i, j] = col


def _render(width, height, time, mouse, device):
    gate = 0.5 + 0.5 * float(np.sin(time * 1.2))
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(fet_kernel, dim=(height, width),
              inputs=[img, float(width / height), float(gate), float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="transistor",
    description="A MOSFET cross-section — source and drain in a silicon substrate, a "
                "gate on a thin oxide. Raise the gate voltage and an inversion channel "
                "forms (ON, current pulses flow); drop it and it vanishes (OFF). "
                "--frames toggles the gate.",
    renderer=_render,
)
