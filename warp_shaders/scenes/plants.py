"""L-System plants that grow — the engine showing life.

Three scenes (``grass`` → ``herb`` → ``tree``, increasing complexity) grown from
L-System grammars, tessellated to a mesh, and ray-cast through the Warp engine
(:mod:`warp_shaders.life`). The generation advances with ``time`` and the camera
is framed to the *fully-grown* plant, so a `--frames`/`--gif` run shows the plant
grow up into a fixed frame from a sprout. iMouse orbits.

    python render.py --scene tree  --frames 8 --fps 6 --gif out/tree.gif
    python render.py --scene grass --time 9 -o grass.png
"""

import math

from ..life import plants as _plants
from ..life.render import render_plant
from ..scene import Scene


def _grow_scene(name: str):
    def _render(width, height, time, mouse, device):
        spec = _plants.get_spec(name)
        # frame to the fully-grown plant so growth rises into a stable view
        _, (lo, hi) = _plants.grow_mesh(spec, spec.gens)
        cx, cy, cz = (float((lo[0] + hi[0]) * 0.5),
                      float(lo[1]), float((lo[2] + hi[2]) * 0.5))
        size = float(max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3))

        g = min(spec.gens, 1 + int(time))            # grow one generation / sec
        mesh, _b = _plants.grow_mesh(spec, g)

        az = 0.6 + time * 0.06 + float(mouse[0]) * 0.01
        dist = size * 1.7 + 2.0
        ty = cy + (hi[1] - lo[1]) * 0.45
        eye = (cx + dist * math.sin(az), ty + size * 0.12 + float(mouse[1]) * 0.02,
               cz + dist * math.cos(az))
        return render_plant(mesh, width, height, eye, (cx, ty, cz),
                            sun_dir=(0.55, 0.82, 0.4), device=device,
                            fov=40.0, exposure=1.05, ground_y=cy)
    return _render


SCENES = [
    Scene(name="grass", renderer=_grow_scene("grass"),
          description="L-System grass tuft growing (Warp mesh ray-cast). --time 0..9."),
    Scene(name="herb", renderer=_grow_scene("herb"),
          description="L-System leafy herb growing (Warp mesh ray-cast). --time 0..6."),
    Scene(name="tree", renderer=_grow_scene("tree"),
          description="L-System tree growing: tapering trunk + leafy canopy. --time 0..7."),
    Scene(name="fern", renderer=_grow_scene("fern"),
          description="L-System bracketed fern unfurling (ABOP fig 1.24, 3D). --time 0..5."),
]
