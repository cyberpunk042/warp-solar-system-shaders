"""DNA double helix — the bottom rung of the "show life" ladder.

A ray-traced B-DNA helix (:func:`warp_shaders.life.molecular.build_helix`) that
**assembles base-pair by base-pair** as ``time`` advances, then turns in place.
Real generated geometry through the same Warp mesh ray-caster the plants use.

    python render.py --scene dna --frames 24 --fps 12 --gif out/dna.gif
    python render.py --scene dna --time 6 -o dna.png
"""

import math

from ..life import molecular as _mol
from ..life.render import render_plant
from ..scene import Scene

_FULL_BP = 20


def _render(width, height, time, mouse, device):
    # assemble the helix over the first ~4s, then keep the full helix turning
    bp = max(2, min(_FULL_BP, 2 + int(time * 5.0)))
    mesh, (lo, hi) = _mol.build_helix(bp=bp, seed=7)
    _, (flo, fhi) = _mol.build_helix(bp=_FULL_BP, seed=7)   # frame to full height
    cx = float((flo[0] + fhi[0]) * 0.5)
    cy = float((flo[1] + fhi[1]) * 0.5)
    cz = float((flo[2] + fhi[2]) * 0.5)
    span = float(fhi[1] - flo[1])

    az = 0.6 + time * 0.5 + float(mouse[0]) * 0.01
    dist = span * 0.9 + 4.0
    eye = (cx + dist * math.sin(az), cy + float(mouse[1]) * 0.02,
           cz + dist * math.cos(az))
    return render_plant(mesh, width, height, eye, (cx, cy, cz),
                        sun_dir=(0.4, 0.7, 0.5), device=device, fov=40.0,
                        exposure=1.06, ground=False)


SCENE = Scene(name="dna", renderer=_render,
              description="DNA double helix assembling base-pair by base-pair "
                          "then turning (Warp mesh ray-cast). --time 0..6.")
