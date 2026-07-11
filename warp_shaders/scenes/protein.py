"""Protein folding — an extended polypeptide collapsing into its fold.

The backbone (:func:`warp_shaders.life.molecular.build_protein`) interpolates
from an extended chain to a compact α-helix→β-strand fold as ``time`` advances,
coloured N→C. Same Warp mesh ray-caster as the plants.

    python render.py --scene protein --frames 24 --fps 12 --gif out/protein.gif
    python render.py --scene protein --time 6 -o protein.png
"""

import math

from ..life import molecular as _mol
from ..life.render import render_plant
from ..scene import Scene


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _render(width, height, time, mouse, device):
    fold = _smooth(time / 5.0)                          # fold up over ~5s
    mesh, (lo, hi) = _mol.build_protein(n=48, fold=fold)
    # frame to the CURRENT extent so the subject fills the frame as it folds
    cx = float((lo[0] + hi[0]) * 0.5)
    cy = float((lo[1] + hi[1]) * 0.5)
    cz = float((lo[2] + hi[2]) * 0.5)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3))

    az = 0.7 + time * 0.25 + float(mouse[0]) * 0.01
    dist = span * 0.95 + 2.5
    eye = (cx + dist * math.sin(az), cy + span * 0.05 + float(mouse[1]) * 0.02,
           cz + dist * math.cos(az))
    return render_plant(mesh, width, height, eye, (cx, cy, cz),
                        sun_dir=(0.45, 0.7, 0.5), device=device, fov=40.0,
                        exposure=1.06, ground=False)


SCENE = Scene(name="protein", renderer=_render,
              description="Polypeptide backbone folding extended->compact, "
                          "coloured N->C (Warp mesh ray-cast). --time 0..6.")
