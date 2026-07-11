"""super-earth — a configurable procedural planet (:mod:`warp_shaders.superearth`).

Every feature is an independent knob, exposed as a family of preset scenes so you
can see the engine build a world piece by piece: a barren rock, an earth-like
globe, an ocean world, a volcanic hell — same kernel, different config.

    python render.py --scene super_earth --frames 24 --fps 12 --gif out/se.gif
    python render.py --scene se_volcanic --time 0 -o volcanic.png
"""

from ..lod import active_tier
from ..scene import Scene
from ..superearth import presets
from ..superearth.moons import moonset
from ..superearth.planet import render_planet


def _scene(cfg_name, moons=None, **kw):
    def _render(width, height, time, mouse, device):
        cfg = presets.get(cfg_name)
        ms = moonset(moons) if moons else []
        return render_planet(cfg, width, height, time, mouse, device,
                             quality=active_tier().name, moons=ms, **kw)
    return _render


SCENES = [
    Scene(name="super_earth", renderer=_scene("earthlike"),
          description="Configurable procedural planet (earth-like preset): "
                      "oceans, continents, mountains, snow, atmosphere."),
    Scene(name="se_barren", renderer=_scene("barren"),
          description="super-earth preset: barren cratered rock, no air/water."),
    Scene(name="se_ocean", renderer=_scene("ocean_world"),
          description="super-earth preset: ocean world with island arcs."),
    Scene(name="se_volcanic", renderer=_scene("volcanic"),
          description="super-earth preset: young volcanic world."),
    Scene(name="se_rivers", renderer=_scene("riverlands"),
          description="super-earth preset: continents laced with rivers + lakes."),
    Scene(name="se_living", renderer=_scene("living", sun_az=3.9, sun_el=0.2),
          description="super-earth preset: a living world — night side aglow with "
                      "bioluminescence + city lights."),
    Scene(name="se_arid", renderer=_scene("arid"),
          description="super-earth preset: arid desert continents (life off)."),
    Scene(name="se_moons", renderer=_scene("earthlike", moons="many",
                                           dist=6.5, fov=52.0),
          description="super-earth with a system of four moons (rocky / icy / "
                      "lava / desert), configurable in number, type and size."),
]
