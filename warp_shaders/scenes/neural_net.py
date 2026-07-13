"""The mind — a neural network firing, signals cascading through the graph.

Nodes (neurons) wired to their neighbours; pulses race along the edges and, when a
node's inputs coincide, it **fires** and flashes its outgoing connections — the
integrate-and-fire traffic that, en masse, is thought. Animate with ``--frames``.
See ``docs/research/24-the-living-body.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene

# --- build a fixed graph once (deterministic) ---
_rng = np.random.RandomState(7)
_N = 26
_ang = _rng.uniform(0, 2 * np.pi, _N)
_rad = np.sqrt(_rng.uniform(0.02, 1.0, _N))
_NPOS = np.stack([_rad * np.cos(_ang) * 1.35, _rad * np.sin(_ang)], 1).astype(np.float32)
_NPH = _rng.uniform(0, 6.28, _N).astype(np.float32)
_edges = []
for a in range(_N):                                    # connect to 3 nearest neighbours
    d = np.linalg.norm(_NPOS - _NPOS[a], axis=1)
    for b in np.argsort(d)[1:4]:
        if (b, a) not in _edges and (a, b) not in _edges:
            _edges.append((a, int(b)))
_EA = np.array([e[0] for e in _edges], np.int32)
_EB = np.array([e[1] for e in _edges], np.int32)
_EPH = _rng.uniform(0, 1.0, len(_edges)).astype(np.float32)


@wp.kernel
def net_kernel(img: wp.array2d(dtype=wp.vec3), npos: wp.array(dtype=wp.vec2),
               nph: wp.array(dtype=float), nn: int, ea: wp.array(dtype=int),
               eb: wp.array(dtype=int), eph: wp.array(dtype=float), ne: int,
               aspect: float, time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.02, 0.025, 0.05)

    # edges: faint wiring + a travelling pulse
    for k in range(ne):
        a = npos[ea[k]]
        b = npos[eb[k]]
        ba = b - a
        pa = p - a
        h = wp.clamp(wp.dot(pa, ba) / wp.max(wp.dot(ba, ba), 1.0e-5), 0.0, 1.0)
        d = wp.length(pa - ba * h)
        wire = wp.exp(-(d / 0.006) * (d / 0.006))
        col = col + wp.vec3(0.2, 0.35, 0.6) * (wire * 0.25)
        s = wp.mod(time * 0.6 + eph[k], 1.0)           # pulse position along the edge
        pp = a + ba * s
        dp = wp.length(p - pp)
        pulse = wp.exp(-(dp / 0.02) * (dp / 0.02))
        col = col + wp.vec3(0.5, 0.85, 1.0) * (pulse * 0.9)

    # nodes: glow, flashing when they fire
    for k in range(nn):
        d = wp.length(p - npos[k])
        fire = wp.pow(wp.max(wp.sin(time * 1.6 + nph[k]), 0.0), 6.0)
        glow = wp.exp(-(d / 0.03) * (d / 0.03)) + 0.25 * wp.exp(-(d / 0.09) * (d / 0.09))
        col = col + wp.vec3(0.6, 0.9, 1.0) * (glow * (0.3 + fire * 1.8))

    img[i, j] = col


def _render(width, height, time, mouse, device):
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(net_kernel, dim=(height, width),
              inputs=[img, wp.array(_NPOS, dtype=wp.vec2, device=device),
                      wp.array(_NPH, dtype=wp.float32, device=device), int(_N),
                      wp.array(_EA, dtype=wp.int32, device=device),
                      wp.array(_EB, dtype=wp.int32, device=device),
                      wp.array(_EPH, dtype=wp.float32, device=device), int(len(_EA)),
                      float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.015))
    hdr = post.bloom(hdr, threshold=0.9, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="neural_net",
    description="The mind — a neural network firing: pulses race along the edges of "
                "a neuron graph and nodes flash as they fire (integrate-and-fire). "
                "--frames animates the cascading thought.",
    renderer=_render,
)
