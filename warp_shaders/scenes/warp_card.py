"""warp_card — fold a flat card into a cube, in time (the 2-D sibling of warp_fold).

Renders the `warp_compress.cardfold` codec as it runs. A flat **card** — a grid of glowing cells,
each coloured by its value — is folded in half like paper; the half that lifts over arcs up and
lands on the other, and wherever the two stacked cells match they **merge** (they go green and dim
into the layer below). Fold after fold, alternating axes, the card halves and thickens, condensing
into a compact **cube** whose bright core is the fundamental tile the whole card was built from.
Run it backwards and the cube unfolds flat again: decompression. What you watch is the real fold
schedule the codec found; the frame is the fold step.
"""

import colorsys

import numpy as np
import warp as wp

from ..engine import post
from ..particles import camera_ray, emitter, orbit_ro
from ..scene import Scene
from warp_compress import cardfold as cf

_CYCLE = 10.0
_CELL = 0.16
_LAYER = 0.12
_ARC = 0.7
_CACHE = {}


def _mirror_expand(a):
    a = np.hstack([a, a[:, ::-1]])
    return np.vstack([a, a[::-1, :]])


def _build(kind):
    if kind in _CACHE:
        return _CACHE[kind]
    rng = np.random.default_rng(5)
    card = rng.integers(0, 256, (4, 4)).astype(np.int32)
    for _ in range(2):                                   # 4x4 -> 8x8 -> 16x16 (nested mirror symmetry)
        card = _mirror_expand(card)
    _core, levels = cf.fold_levels_card(card, tol=0)
    axes = [lvl.axis for lvl in levels]
    h, w = card.shape

    # simulate the fold schedule, recording every cell's (px, py, layer) after each fold
    rr, cc = np.mgrid[0:h, 0:w]
    px = (cc - (w - 1) / 2.0).astype(np.float64).ravel()
    py = (rr - (h - 1) / 2.0).astype(np.float64).ravel()
    layer = np.zeros(px.size, np.float64)
    ex, ey = w / 2.0, h / 2.0
    states = [(px.copy(), py.copy(), layer.copy())]
    for a in axes:
        if a == 1:
            far = px > 0
            px[far] = -px[far]
            layer[far] += 1
            px += ex / 2.0
            ex /= 2.0
        else:
            far = py > 0
            py[far] = -py[far]
            layer[far] += 1
            py += ey / 2.0
            ey /= 2.0
        states.append((px.copy(), py.copy(), layer.copy()))

    vals = card.ravel()
    uniq = sorted(set(int(v) for v in vals))
    pal = {v: colorsys.hsv_to_rgb((k / max(1, len(uniq))) % 1.0, 0.8, 1.0)
           for k, v in enumerate(uniq)}
    base_col = np.array([pal[int(v)] for v in vals], np.float32)

    out = (states, base_col, len(axes))
    _CACHE[kind] = out
    return out


def _world(state):
    px, py, layer = state
    p = np.zeros((px.size, 3), np.float32)
    p[:, 0] = px * _CELL
    p[:, 2] = py * _CELL
    p[:, 1] = layer * _LAYER
    return p


def _smooth(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _layout(kind, g):
    states, base_col, F = _build(kind)
    if F == 0:
        pos = _world(states[0]); return pos, base_col.copy(), np.full(len(pos), 0.09, np.float32)
    gg = float(np.clip(g, 0.0, 1.0)) * F
    k = min(int(gg), F - 1)
    frac = gg - k
    s = _smooth(frac)
    a = _world(states[k])
    b = _world(states[k + 1])
    pos = a * (1.0 - s) + b * s
    moved = states[k + 1][2] > states[k][2]              # cells that lift over on this fold
    pos[:, 1] = pos[:, 1] + np.sin(frac * np.pi) * _ARC * moved
    layer = states[k][2] * (1.0 - s) + states[k + 1][2] * s

    green = np.array([0.2, 1.0, 0.35], np.float32)
    t = np.clip(layer[:, None] / 2.5, 0.0, 0.8)
    col = (base_col * (1.0 - t) + green * t).astype(np.float32)
    col *= (1.0 - 0.35 * np.clip(layer[:, None] / 4.0, 0, 1)).astype(np.float32)  # deeper layers dim
    siz = (0.085 - 0.008 * layer).astype(np.float32)
    return pos, col, np.clip(siz, 0.04, 0.09)


@wp.kernel
def _kernel(img: wp.array2d(dtype=wp.vec3), width: int, height: int, time: float, mouse: wp.vec2,
            pos: wp.array(dtype=wp.vec3), col: wp.array(dtype=wp.vec3),
            siz: wp.array(dtype=wp.float32), count: int):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))
    ro = orbit_ro(time, mouse, res, 6.2)
    uvx = ((float(j) + 0.5) - 0.5 * res[0]) / res[1]
    uvy = ((float(height - 1 - i) + 0.5) - 0.5 * res[1]) / res[1]
    rd = camera_ray(wp.vec2(uvx, uvy), ro, wp.vec3(0.0, 0.0, 0.0), 1.5)
    c = wp.vec3(0.02, 0.03, 0.06)
    for k in range(count):
        c = c + col[k] * emitter(ro, rd, pos[k], siz[k])
    img[i, j] = c


def _progress(time):
    u = (float(time) % _CYCLE) / _CYCLE
    return 1.0 - abs(2.0 * u - 1.0)


def _make(kind):
    def _render(width, height, time, mouse, device):
        pos, col, siz = _layout(kind, _progress(time))
        img = wp.zeros((height, width), dtype=wp.vec3, device=device)
        wp.launch(_kernel, dim=(height, width),
                  inputs=[img, width, height, float(time), mouse,
                          wp.array(pos, dtype=wp.vec3, device=device),
                          wp.array(col, dtype=wp.vec3, device=device),
                          wp.array(siz, dtype=wp.float32, device=device), len(pos)],
                  device=device)
        wp.synchronize_device(device)
        return post.tonemap(img.numpy(), mode="aces", exposure=1.2, preserve_hue=True)
    return _render


SCENES = [
    Scene(name="warp_card",
          description="fold a flat card into a cube, in time — the warp_compress card codec folds "
                      "the grid in half again and again, matching cells flashing green and merging "
                      "into the layer below, condensing to a bright cube core, then unfolding flat "
                      "on decompression.",
          renderer=_make("card")),
]
