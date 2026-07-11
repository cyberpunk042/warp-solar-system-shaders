"""The mind — a plant that *chooses* whether to obey the obvious rules.

A little Conway's-Life mind (:class:`warp_shaders.life.mind.Mind`) deliberates;
its living population sets a **drive** in [0, 1]. That drive steers the same
tropism knobs the reflex scenes use — but now as a *decision*:

- **high drive** (active mind) ⇒ *seek the light*: open leaves, strong phototropism.
- **low drive** (quiescent) ⇒ *rest*: the plant sags and folds its leaves shut.

So unlike ``phototropism`` (which *always* follows the light), here the plant
turns toward the light only when the mind decides to — and closes up when it
doesn't. The CA is drawn as an inset panel with a drive bar, so you can watch the
deliberation and its effect at once.

    python render.py --scene mind --frames 48 --fps 12 --gif out/mind.gif
    python render.py --scene mind --time 4 -o mind.png
"""

import math
from dataclasses import replace

import numpy as np

from ..life import mind as _mind
from ..life import plants as _plants
from ..life.render import render_plant
from ..scene import Scene

_GRID = 40
_SEED = 3
_LIGHT = (8.0, 6.0, 2.5)


def _paint_panel(img, grid, drive):
    """Draw the CA grid + a drive bar into the top-left corner of `img`."""
    h, w, _ = img.shape
    panel = int(min(h, w) * 0.30)
    m = max(4, int(min(h, w) * 0.02))
    up = np.kron(grid, np.ones((max(panel // grid.shape[0], 1),) * 2, np.uint8))
    up = up[:panel, :panel]
    ph, pw = up.shape
    live = np.array([0.35, 1.0, 0.85], np.float32)
    dead = np.array([0.04, 0.06, 0.10], np.float32)
    block = np.where(up[..., None] > 0, live, dead).astype(np.float32)
    img[m:m + ph, m:m + pw] = block
    # 2px border
    b = np.array([0.6, 0.7, 0.8], np.float32)
    img[m - 2:m, m - 2:m + pw + 2] = b
    img[m + ph:m + ph + 2, m - 2:m + pw + 2] = b
    img[m - 2:m + ph + 2, m - 2:m] = b
    img[m - 2:m + ph + 2, m + pw:m + pw + 2] = b
    # drive bar under the panel: length ∝ drive; green=seek light, amber=rest
    by = m + ph + m // 2
    bh = max(3, m // 2)
    fill = int(pw * min(max(drive, 0.0), 1.0))
    col = np.array([0.3, 0.9, 0.4], np.float32) if drive >= 0.5 \
        else np.array([0.95, 0.7, 0.25], np.float32)
    img[by:by + bh, m:m + pw] = np.array([0.10, 0.12, 0.16], np.float32)
    if fill > 0:
        img[by:by + bh, m:m + fill] = col
    return img


def _render(width, height, time, mouse, device):
    steps = int(time * 8.0)
    mind = _mind.run_to(_GRID, _SEED, steps)
    drive = mind.decision()

    spec = _plants.get_spec("sapling")
    cfg = replace(spec.cfg,
                  light=_LIGHT, light_e=0.14 * drive,
                  tropism=(0.0, -1.0, 0.0), tropism_e=0.13 * (1.0 - drive),
                  leaf_fold=0.75 * (1.0 - drive))
    mesh, (lo, hi) = _plants.grow_mesh_env(spec, spec.gens, cfg)

    cx = float((lo[0] + hi[0]) * 0.5)
    cy = float(lo[1])
    cz = float((lo[2] + hi[2]) * 0.5)
    size = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3))
    az = 0.5 + time * 0.04 + float(mouse[0]) * 0.01
    dist = size * 1.9 + 2.5
    ty = cy + (hi[1] - lo[1]) * 0.5
    eye = (cx + dist * math.sin(az), ty + size * 0.12 + float(mouse[1]) * 0.02,
           cz + dist * math.cos(az))
    # sun sits toward the light the mind may choose to follow
    img = render_plant(mesh, width, height, eye, (cx, ty, cz),
                       sun_dir=(0.55, 0.8, 0.4), device=device, fov=42.0,
                       exposure=1.06, ground_y=cy)
    return _paint_panel(img, mind.grid, drive)


SCENE = Scene(name="mind", renderer=_render,
              description="A Conway-Life 'mind' choosing when the plant seeks "
                          "light vs rests/closes (inset grid + drive bar). "
                          "--frames 48 --fps 12.")
