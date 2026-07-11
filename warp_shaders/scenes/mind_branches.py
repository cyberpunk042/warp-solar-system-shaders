"""A per-branch mind — the plant closing *pieces of itself*, independently.

Where the ``mind`` scene makes the *whole* plant choose, here one plant has
several shoots and **each shoot is steered by a different region of the same
Conway mind** (`Mind.decisions(k)` splits the grid into vertical bands). So one
shoot reaches up and opens toward the light while its neighbour — reading a
quieter part of the mind — sags and folds shut. This is the operator's *"close
piece of itself"*: the decision is per-part, not global.

Each shoot is the ``sapling`` grammar grown with its own per-frame
``TurtleConfig`` (light gain / gravity / leaf-fold from its band's drive),
rotated out from a common base and merged into one mesh. The inset shows the
grid split into colour-tinted bands with one drive bar per shoot.

    python render.py --scene mind_branches --frames 48 --fps 12 --gif out/mb.gif
    python render.py --scene mind_branches --time 4 -o mb.png
"""

import math
from dataclasses import replace

import numpy as np

from ..life import mind as _mind
from ..life import plants as _plants
from ..life.lsystem import parse
from ..life.mesh import build_mesh, merge_meshes
from ..life.render import render_plant
from ..life.turtle import interpret
from ..scene import Scene

_GRID = 40
_SEED = 5
_K = 5                       # shoots (and mind bands)
_LIGHT = (1.5, 9.0, 1.0)     # high light: reaching shoots straighten + open
_BAND_TINT = np.array([[0.35, 1.0, 0.85], [1.0, 0.85, 0.4], [0.6, 0.8, 1.0],
                       [1.0, 0.6, 0.8], [0.7, 1.0, 0.55]], np.float32)


def _shoot_mesh(spec, gen, cfg, az_deg, pitch_deg):
    # grow one shoot, rotated out of the common base by (azimuth, pitch)
    word = parse(f"/({az_deg:g})&({pitch_deg:g})") + _plants.derive_word(spec, gen)
    geo = interpret(word, cfg)
    return build_mesh(geo, sides=spec.sides)


def _cfg_for(spec, drive):
    return replace(spec.cfg, light=_LIGHT, light_e=0.15 * drive,
                   tropism=(0.0, -1.0, 0.0), tropism_e=0.13 * (1.0 - drive),
                   leaf_fold=0.8 * (1.0 - drive))


def _paint_panel(img, grid, drives):
    h, w, _ = img.shape
    panel = int(min(h, w) * 0.30)
    m = max(4, int(min(h, w) * 0.02))
    k = len(drives)
    rep = max(panel // grid.shape[0], 1)
    up = np.kron(grid, np.ones((rep, rep), np.uint8))[:panel, :panel]
    ph, pw = up.shape
    dead = np.array([0.04, 0.06, 0.10], np.float32)
    block = np.empty((ph, pw, 3), np.float32)
    for bi in range(k):                       # tint each vertical band
        c0 = bi * pw // k
        c1 = (bi + 1) * pw // k
        seg = up[:, c0:c1]
        block[:, c0:c1] = np.where(seg[..., None] > 0,
                                   _BAND_TINT[bi % len(_BAND_TINT)], dead)
    img[m:m + ph, m:m + pw] = block
    b = np.array([0.6, 0.7, 0.8], np.float32)
    img[m - 2:m, m - 2:m + pw + 2] = b
    img[m + ph:m + ph + 2, m - 2:m + pw + 2] = b
    img[m - 2:m + ph + 2, m - 2:m] = b
    img[m - 2:m + ph + 2, m + pw:m + pw + 2] = b
    # one drive bar per shoot, under the panel, coloured by that band's tint
    bh = max(3, m // 2)
    for bi in range(k):
        by = m + ph + m // 2 + bi * (bh + 2)
        fill = int(pw * min(max(drives[bi], 0.0), 1.0))
        img[by:by + bh, m:m + pw] = np.array([0.10, 0.12, 0.16], np.float32)
        if fill > 0:
            img[by:by + bh, m:m + fill] = _BAND_TINT[bi % len(_BAND_TINT)]
    return img


def _render(width, height, time, mouse, device):
    mind = _mind.run_to(_GRID, _SEED, int(time * 8.0))
    drives = mind.decisions(_K)

    spec = _plants.get_spec("sapling")
    meshes = []
    for bi in range(_K):
        az = bi * (360.0 / _K)
        meshes.append(_shoot_mesh(spec, spec.gens, _cfg_for(spec, drives[bi]),
                                  az, 40.0))
    plant = merge_meshes(meshes)
    lo, hi = plant.verts.min(0), plant.verts.max(0)

    cx = float((lo[0] + hi[0]) * 0.5)
    cy = float(lo[1])
    cz = float((lo[2] + hi[2]) * 0.5)
    size = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3))
    az = 0.5 + time * 0.04 + float(mouse[0]) * 0.01
    dist = size * 1.7 + 2.5
    ty = cy + (hi[1] - lo[1]) * 0.5
    eye = (cx + dist * math.sin(az), ty + size * 0.14 + float(mouse[1]) * 0.02,
           cz + dist * math.cos(az))
    img = render_plant(plant, width, height, eye, (cx, ty, cz),
                       sun_dir=(0.5, 0.82, 0.4), device=device, fov=44.0,
                       exposure=1.06, ground_y=cy)
    return _paint_panel(img, mind.grid, drives)


SCENE = Scene(name="mind_branches", renderer=_render,
              description="Per-branch mind: each shoot of one plant is steered by "
                          "a different region of the Conway grid — some reach for "
                          "light, some close. --frames 48 --fps 12.")
