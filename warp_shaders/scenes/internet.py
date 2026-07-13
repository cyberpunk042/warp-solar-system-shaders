"""The internet — packets hopping across a router mesh.

Computers talk by chopping messages into **packets** that hop, router to router,
across a mesh network — each packet independently routed and reassembled (TCP/IP).
Here a graph of **routers** (glowing nodes) wired into a mesh, with bright
**packets** streaming along the edges from node to node. See
``docs/research/26-the-machine.md``. --frames animates the traffic.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene

# ---- a deterministic router mesh: 12 nodes, each wired to its 3 nearest ----
_NODES = [
    (-1.15, 0.55), (-0.78, -0.08), (-1.05, -0.62),
    (-0.38, 0.66), (-0.18, 0.04), (-0.52, -0.55),
    (0.28, 0.52), (0.42, -0.06), (0.06, -0.62),
    (0.98, 0.56), (0.86, -0.22), (0.56, -0.60),
]


def _build_edges():
    n = len(_NODES)
    es = set()
    for a in range(n):
        d = sorted(range(n), key=lambda b: (_NODES[a][0] - _NODES[b][0]) ** 2
                   + (_NODES[a][1] - _NODES[b][1]) ** 2)
        for b in d[1:4]:                         # 3 nearest neighbours
            es.add((min(a, b), max(a, b)))
    return sorted(es)


_EDGES = _build_edges()


@wp.func
def _seg(p: wp.vec2, a: wp.vec2, b: wp.vec2) -> float:
    pa = p - a
    ba = b - a
    h = wp.clamp(wp.dot(pa, ba) / wp.dot(ba, ba), 0.0, 1.0)
    return wp.length(pa - ba * h)


@wp.func
def _glow(d: float, w: float) -> float:
    return wp.exp(-(d / w) * (d / w))


@wp.kernel
def net_kernel(img: wp.array2d(dtype=wp.vec3), nodes: wp.array(dtype=wp.vec2),
               ea: wp.array(dtype=wp.int32), eb: wp.array(dtype=wp.int32),
               seed: wp.array(dtype=wp.float32), ne: int, nn: int,
               aspect: float, time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0
    p = wp.vec2(x, y)
    col = wp.vec3(0.015, 0.02, 0.035)

    # edges + travelling packets
    for e in range(ne):
        a = nodes[ea[e]]
        b = nodes[eb[e]]
        d = _seg(p, a, b)
        col = col + wp.vec3(0.15, 0.32, 0.5) * _glow(d, 0.006) * 0.7
        s = seed[e]
        # 1-2 packets per edge, each a bright dot sliding a->b (some reversed)
        for k in range(2):
            ph = wp.mod(time * (0.25 + 0.12 * s) + s * 3.1 + float(k) * 0.5, 1.0)
            dir_ = wp.step(0.5 - wp.mod(s * 7.0, 1.0))     # some edges flow b->a
            f = ph * dir_ + (1.0 - ph) * (1.0 - dir_)
            pk = a + (b - a) * f
            col = col + wp.vec3(0.7, 0.95, 1.0) * _glow(wp.length(p - pk), 0.014) * 2.0

    # router nodes: glowing hubs
    for m in range(nn):
        dn = wp.length(p - nodes[m])
        col = col + wp.vec3(0.5, 0.85, 1.0) * _glow(dn, 0.03) * 1.3
        col = col + wp.vec3(1.0, 1.0, 1.0) * _glow(dn, 0.01) * 1.2

    img[i, j] = col


def _render(width, height, time, mouse, device):
    nodes = wp.array(np.array(_NODES, dtype=np.float32), dtype=wp.vec2, device=device)
    ea = wp.array(np.array([e[0] for e in _EDGES], dtype=np.int32), dtype=wp.int32, device=device)
    eb = wp.array(np.array([e[1] for e in _EDGES], dtype=np.int32), dtype=wp.int32, device=device)
    seed = wp.array(np.array([(math.sin(k * 12.9898) * 43758.5453) % 1.0
                              for k in range(len(_EDGES))], dtype=np.float32),
                    dtype=wp.float32, device=device)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(net_kernel, dim=(height, width),
              inputs=[img, nodes, ea, eb, seed, len(_EDGES), len(_NODES),
                      float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.012))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.5, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="internet",
    description="The internet as a router mesh — glowing hubs wired into a graph with "
                "bright packets streaming along the edges, router to router, each "
                "independently routed (TCP/IP). --frames animates the traffic.",
    renderer=_render,
)
