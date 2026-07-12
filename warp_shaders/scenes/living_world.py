"""A living world — the L-System ecosystem under its sun, over a day.

Fuses the three strands (engine + solar system + life): the `ecosystem` meadow
grown on a surface, lit by a single **sun that arcs across the sky** — low and
amber at dawn (long shadows), white and high at noon, amber again at dusk. The
plants are **phototropic**, so the whole meadow leans and follows the sun through
the day. `time` is the **day fraction** (0 = dawn, 0.5 = noon, 1 = dusk). See
``docs/research/16-a-living-world.md``.

    python render.py --scene living_world --time 0.5 -o noon.png
    python render.py --scene living_world --frames 48 --fps 12 --gif out/day.gif
"""

import math

from dataclasses import replace

from ..engine.color import kelvin_to_rgb_np
from ..life import ecosystem as _eco
from ..life import plants as _plants
from ..life.mesh import merge_meshes
from ..life.render import render_world
from ..scene import Scene

_ECO = _eco.Ecosystem(seed=7, pool=95, radius=11.0)     # a dense, wide meadow


def _sun(day):
    """Sun direction, colour and intensity for a day fraction (0=dawn, 1=dusk).
    The sun crosses the **front** sky (toward -z) so the camera faces it."""
    ang = day * math.pi                          # 0..pi across the sky
    el = math.sin(ang) * 0.95 + 0.05             # elevation: low -> high -> low
    d = (0.8 * math.cos(ang), el, -0.6)          # east -> up -> west, in front
    day_h = max(math.sin(ang), 0.0)              # 0 at horizon, 1 at noon
    temp = 2200.0 + 3500.0 * day_h               # amber at horizon -> white at noon
    col = kelvin_to_rgb_np(temp).tolist()
    inten = 0.5 + 0.75 * day_h
    return d, col, inten


def _render(width, height, time, mouse, device):
    day = max(0.02, min(time, 0.98))
    phase = 0.16                                 # a fixed lush high-summer meadow
    pal = _eco.season_palette(phase)
    vig = _eco.vigor(phase)
    sdir, scol, sint = _sun(day)

    meshes, offsets = [], []
    for st in _ECO.standing(2.0 + phase):        # a settled, grown meadow
        spec = _plants.get_spec(st.plant.species)
        # phototropism: lean toward the sun's horizontal direction
        light_pt = (st.plant.x + sdir[0] * 6.0, 6.0, st.plant.z + sdir[2] * 6.0)
        cfg = replace(spec.cfg, palette=pal,
                      leaf_size=spec.cfg.leaf_size * (0.5 + 0.5 * vig),
                      light=light_pt, light_e=0.10,
                      tropism=(0.0, 1.0, 0.0), tropism_e=0.02)
        mesh, _b = _plants.grow_mesh_env(spec, st.gen, cfg)
        if mesh.n_tris:
            meshes.append(mesh)
            offsets.append((st.plant.x, 0.0, st.plant.z))
    field = merge_meshes(meshes, offsets)

    # low, immersive camera facing the sun across the meadow (long shadows)
    dist = _ECO.radius * 1.1 + 5.0
    eye = (float(mouse[0]) * 0.02, 0.7, dist)
    return render_world(field, width, height, eye, (0.0, 1.5, -2.0),
                        [(sdir, scol, sint)], device=device, fov=58.0,
                        exposure=1.08, ground_col=(0.09, 0.22, 0.08))


SCENE = Scene(name="living_world", renderer=_render,
              description="The L-System ecosystem under its sun over a day — sun "
                          "arcs dawn->noon->dusk, phototropic plants follow it, "
                          "long amber shadows. --time is the day fraction 0..1.")
