"""Wave and collapse — a plant's possible futures resolving to one.

Several *different* plant futures (herb, bush, flower, fern, sapling), all rooted
at the same spot, begin **superposed** — faint overlapping ghosts, a cloud of
what the plant might become. Over ``time`` the superposition **collapses** to a
single realised plant, the collapse front sweeping tip→base (the future settling
first, reaching *backward* into its own history). A Conway **mind**
(:mod:`warp_shaders.life.mind`) biases *which* future resolves.

An explicit metaphor for the operator's *"things are waves before … a collapse in
the world"* — see [Research 07](../research/07-wave-and-collapse.md). The rendered
ensemble is cached (fixed camera), so only the cheap per-frame blend re-runs.

    python render.py --scene wavecollapse --frames 40 --fps 10 --gif out/wc.gif
    python render.py --scene wavecollapse --time 5 -o wc.png
"""

import math

import numpy as np

from ..life import collapse as _c
from ..life import mind as _mind
from ..life import plants as _plants
from ..life.render import render_plant
from ..scene import Scene

_GRID = 40
_SEED = 4
_T = 5.0                       # collapse duration (seconds)
_RATE = 6.0                    # mind steps per second
_DECISION_STEP = 8             # mind state that fixes which future wins
# (factory, seed) — five visibly-distinct possible futures
_FUTURES = [("herb", 3), ("bush", 6), ("flower", 0), ("fern", 4), ("sapling", 5)]
_GHOSTS: dict = {}


def _ensemble():
    specs = []
    for name, seed in _FUTURES:
        fn = _plants._REGISTRY[name]
        try:
            specs.append(fn(seed=seed))
        except TypeError:
            specs.append(fn())          # flower() takes no seed
    return specs


def _render_ghosts(width, height, device):
    key = (width, height, device)
    if key in _GHOSTS:
        return _GHOSTS[key]
    specs = _ensemble()
    meshes, los, his = [], [], []
    for s in specs:
        m, (lo, hi) = _plants.grow_mesh(s, s.gens)
        meshes.append(m); los.append(lo); his.append(hi)
    lo = np.min(np.array(los), 0); hi = np.max(np.array(his), 0)
    cx, cz = float((lo[0] + hi[0]) * 0.5), float((lo[2] + hi[2]) * 0.5)
    cy = float(lo[1]); size = float(max(hi[0] - lo[0], hi[1] - lo[1],
                                        hi[2] - lo[2], 1e-3))
    ty = cy + (hi[1] - lo[1]) * 0.5
    dist = size * 1.7 + 2.5
    eye = (cx + dist * 0.5, ty + size * 0.15, cz + dist * 0.87)
    imgs = [render_plant(m, width, height, eye, (cx, ty, cz),
                         sun_dir=(0.5, 0.82, 0.4), device=device, fov=44.0,
                         exposure=1.06, ground_y=cy) for m in meshes]
    # find the plants' vertical pixel extent (rows differing from the top sky
    # row) so the collapse front sweeps the subject, not the empty sky above it
    cloud = np.stack(imgs, 0).mean(0)
    sky = cloud[0].mean(0)
    rowmax = np.abs(cloud - sky).max(2).max(1)
    content = np.where(rowmax > 0.03)[0]
    if content.size:
        rr = (float(max(content.min() - 4, 0)), float(min(content.max() + 4, height)))
    else:
        rr = (0.0, float(height))
    _GHOSTS[key] = (imgs, rr)
    return _GHOSTS[key]


def _panel(img, grid, n, chosen):
    h, w, _ = img.shape
    panel = int(min(h, w) * 0.26)
    m = max(4, int(min(h, w) * 0.02))
    rep = max(panel // grid.shape[0], 1)
    up = np.kron(grid, np.ones((rep, rep), np.uint8))[:panel, :panel]
    ph, pw = up.shape
    live = np.array([0.6, 0.85, 1.0], np.float32)
    dead = np.array([0.05, 0.06, 0.10], np.float32)
    img[m:m + ph, m:m + pw] = np.where(up[..., None] > 0, live, dead)
    b = np.array([0.6, 0.7, 0.8], np.float32)
    img[m - 2:m, m - 2:m + pw + 2] = b; img[m + ph:m + ph + 2, m - 2:m + pw + 2] = b
    img[m - 2:m + ph + 2, m - 2:m] = b; img[m - 2:m + ph + 2, m + pw:m + pw + 2] = b
    # a row of N swatches; the chosen future glows, the rest are dim
    sw = pw // n
    sy = m + ph + m // 2
    sh = max(4, m // 2)
    for i in range(n):
        c = np.array([0.4, 1.0, 0.5], np.float32) if i == chosen \
            else np.array([0.25, 0.28, 0.34], np.float32)
        img[sy:sy + sh, m + i * sw:m + (i + 1) * sw - 1] = c
    return img


def _render(width, height, time, mouse, device):
    ghosts, row_range = _render_ghosts(width, height, device)
    n = len(ghosts)
    # which future the mind favours — fixed at the decision moment (stable)
    chosen = _c.pick_index(_mind.run_to(_GRID, _SEED, _DECISION_STEP).decisions(n))
    front = time / _T
    front = front * front * (3.0 - 2.0 * front) if front < 1.0 else 1.0
    img = _c.collapse_blend(ghosts, chosen, front, row_range=row_range)
    live = _mind.run_to(_GRID, _SEED, int(time * _RATE))     # live grid for flavor
    return _panel(img.copy(), live.grid, n, chosen)


SCENE = Scene(name="wavecollapse", renderer=_render,
              description="Superposed possible plant futures collapsing to one "
                          "(mind-biased), the front sweeping tip->base. --time 0..5.")
