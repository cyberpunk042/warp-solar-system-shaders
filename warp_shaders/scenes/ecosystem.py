"""A living meadow over the seasons — life at the population scale.

A patch of L-System plants (:mod:`warp_shaders.life.ecosystem`) that live over
**years**: born, growing, blooming, senescing, dying, with new seedlings filling
the gaps. The meadow **recolours with the seasons** and the plants **compete for
light** — a shaded plant grows less and leans toward the open sky. Everything is
grown through the usual L-System pipeline and merged into one ray-cast mesh.

``time`` is in **years** (1.0 = a full spring→winter cycle).

    python render.py --scene ecosystem --frames 60 --fps 12 --gif out/eco.gif
    python render.py --scene ecosystem --time 0.35 -o summer.png
"""

import math
from dataclasses import replace

from ..life import ecosystem as _eco
from ..life import plants as _plants
from ..life.mesh import merge_meshes
from ..life.render import render_plant
from ..scene import Scene

_ECO = _eco.Ecosystem(seed=7)


def _sun_for(phase):
    # summer high + bright, autumn warm + lower, winter low + cool
    if phase < 0.5:
        return (0.5, 0.82, 0.4), 1.07
    if phase < 0.75:
        return (0.55, 0.6, 0.45), 1.04          # autumn: lower, warm
    return (0.6, 0.42, 0.5), 0.96               # winter: low, dim


def _render(width, height, time, mouse, device):
    phase = _eco.season_phase(time)
    pal = _eco.season_palette(phase)
    vig = _eco.vigor(phase)
    gust = 0.04 + 0.04 * math.sin(time * 9.0)   # a light breeze
    wx, wz = math.cos(0.6), math.sin(0.6)

    meshes, offsets = [], []
    for st in _ECO.standing(time):
        spec = _plants.get_spec(st.plant.species)
        light_pt, light_e = None, 0.0
        if st.lean is not None:
            light_pt = (st.plant.x + st.lean[0] * 6.0, 7.0,
                        st.plant.z + st.lean[1] * 6.0)
            light_e = 0.10 * (1.0 - st.light)   # shaded plants lean harder
        cfg = replace(spec.cfg, palette=pal,
                      leaf_size=spec.cfg.leaf_size * (0.45 + 0.55 * vig),
                      light=light_pt, light_e=light_e,
                      tropism=(wx, -0.15, wz), tropism_e=max(gust, 0.0))
        mesh, _b = _plants.grow_mesh_env(spec, st.gen, cfg)
        if mesh.n_tris:
            meshes.append(mesh)
            offsets.append((st.plant.x, 0.0, st.plant.z))
    field = merge_meshes(meshes, offsets)

    sun, expo = _sun_for(phase)
    az = 0.4 + time * 0.25 + float(mouse[0]) * 0.01
    dist = _ECO.radius * 2.1 + 4.0
    eye = (dist * math.sin(az), _ECO.radius * 0.9 + float(mouse[1]) * 0.02,
           dist * math.cos(az))
    return render_plant(field, width, height, eye, (0.0, 1.2, 0.0),
                        sun_dir=sun, device=device, fov=52.0, exposure=expo,
                        ground_y=0.0)


SCENE = Scene(name="ecosystem", renderer=_render,
              description="A living meadow over the seasons: plants born/grow/"
                          "bloom/senesce/die + reseed, competing for light, "
                          "recolouring spring->winter. --time in years, 0..4.")
