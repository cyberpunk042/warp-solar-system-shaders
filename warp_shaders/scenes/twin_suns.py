"""Twin suns — the living world under a Kepler-16-like binary ("Tatooine").

The same ecosystem meadow, but lit by **two suns**: a warm orange **K-dwarf** and
a cooler white companion, low over the horizon — a habitable take on the
circumbinary system Kepler-16, where you would see two suns and two sunsets. Each
sun casts its **own shadow** (offset by their angular separation) tinted by the
*other* sun's colour, and a plant in one sun's shadow is still lit by the other —
soft, coloured, doubled shadows. `time` slowly separates the pair toward a double
sunset. See ``docs/research/16-a-living-world.md``.

    python render.py --scene twin_suns -o tatooine.png
    python render.py --scene twin_suns --frames 48 --fps 12 --gif out/two_suns.gif
"""

import math

from dataclasses import replace

from ..engine.color import kelvin_to_rgb_np
from ..life import ecosystem as _eco
from ..life import plants as _plants
from ..life.mesh import merge_meshes
from ..life.render import render_world
from ..scene import Scene

_ECO = _eco.Ecosystem(seed=7, pool=95, radius=11.0)


def _suns(time):
    """Two low suns drifting apart toward a double sunset over `time` (0..1)."""
    drop = 0.10 * time
    # K-dwarf: warm orange, the brighter primary, low over the horizon
    aA = -0.35 - 0.25 * time
    dA = (0.8 * math.sin(aA), 0.18 - drop, -0.62)
    cA = kelvin_to_rgb_np(2900.0).tolist()
    # companion: cooler white, dimmer, the other side
    aB = 0.30 + 0.30 * time
    dB = (0.8 * math.sin(aB), 0.24 - drop, -0.70)
    cB = kelvin_to_rgb_np(6400.0).tolist()
    return [(dA, cA, 1.05), (dB, cB, 0.6)]


def _render(width, height, time, mouse, device):
    day = max(0.0, min(time, 1.0))
    phase = 0.16
    pal = _eco.season_palette(phase)
    vig = _eco.vigor(phase)
    suns = _suns(day)
    # plants lean toward the brighter (primary) sun
    sdir = suns[0][0]

    meshes, offsets = [], []
    for st in _ECO.standing(2.0 + phase):
        spec = _plants.get_spec(st.plant.species)
        light_pt = (st.plant.x + sdir[0] * 6.0, 6.0, st.plant.z + sdir[2] * 6.0)
        cfg = replace(spec.cfg, palette=pal,
                      leaf_size=spec.cfg.leaf_size * (0.5 + 0.5 * vig),
                      light=light_pt, light_e=0.09,
                      tropism=(0.0, 1.0, 0.0), tropism_e=0.02)
        mesh, _b = _plants.grow_mesh_env(spec, st.gen, cfg)
        if mesh.n_tris:
            meshes.append(mesh)
            offsets.append((st.plant.x, 0.0, st.plant.z))
    field = merge_meshes(meshes, offsets)

    dist = _ECO.radius * 1.1 + 5.0
    eye = (float(mouse[0]) * 0.02, 0.7, dist)
    return render_world(field, width, height, eye, (0.0, 1.5, -2.0), suns,
                        device=device, fov=58.0, exposure=1.06,
                        ground_col=(0.09, 0.21, 0.08))


SCENE = Scene(name="twin_suns", renderer=_render,
              description="The living meadow under a Kepler-16-like binary — a warm "
                          "K-dwarf + a cool companion, two coloured shadows, drifting "
                          "to a double sunset. --frames for the two suns setting.")
