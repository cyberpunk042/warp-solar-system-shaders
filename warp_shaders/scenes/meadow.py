"""A meadow — a little ecosystem of L-System plants swaying in one wind.

Several different plants (grass, herb, flower, fern, bush) are grown, placed on a
ground patch, and **merged into one** ``wp.Mesh``
(:func:`warp_shaders.life.mesh.merge_meshes`) so they ray-cast together. All of
them bend under a **shared, time-varying wind** (the tropism layer), so the whole
field ripples as one — the middle of the "show life" ladder, a habitat rather
than a specimen.

    python render.py --scene meadow --frames 48 --fps 16 --gif out/meadow.gif
    python render.py --scene meadow --time 3 -o meadow.png
"""

import math
from dataclasses import replace

from ..life import plants as _plants
from ..life.mesh import merge_meshes
from ..life.render import render_plant
from ..scene import Scene

# (species, x, z, generation) — a small hand-placed arrangement
_PLOT = [
    ("bush",   -2.6, -0.6, 5),
    ("fern",    2.4, -0.8, 5),
    ("flower", -0.4,  0.9, 6),
    ("flower",  1.1,  1.4, 6),
    ("herb",    0.6, -1.2, 6),
    ("grass",  -1.4,  0.3, 9),
    ("grass",   1.9,  0.4, 9),
    ("grass",   0.1,  1.9, 9),
    ("grass",  -2.2,  1.5, 9),
]


def _render(width, height, time, mouse, device):
    gust = 0.045 + 0.045 * math.sin(time * 1.4) + 0.02 * math.sin(time * 3.7)
    wx, wz = math.cos(0.6), math.sin(0.6)

    meshes, offsets = [], []
    for name, x, z, gen in _PLOT:
        spec = _plants.get_spec(name)
        cfg = replace(spec.cfg, tropism=(wx, -0.12, wz), tropism_e=max(gust, 0.0))
        mesh, _b = _plants.grow_mesh_env(spec, gen, cfg)
        meshes.append(mesh)
        offsets.append((x, 0.0, z))
    field = merge_meshes(meshes, offsets)

    az = 0.5 + time * 0.05 + float(mouse[0]) * 0.01
    dist = 9.5
    eye = (dist * math.sin(az), 3.2 + float(mouse[1]) * 0.02, dist * math.cos(az))
    return render_plant(field, width, height, eye, (0.0, 1.4, 0.2),
                        sun_dir=(0.5, 0.78, 0.42), device=device, fov=46.0,
                        exposure=1.06, ground_y=0.0)


SCENE = Scene(name="meadow", renderer=_render,
              description="A meadow of L-System plants (grass/herb/flower/fern/"
                          "bush) merged into one mesh, swaying in one wind.")
