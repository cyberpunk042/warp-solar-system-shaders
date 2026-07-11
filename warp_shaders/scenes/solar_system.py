"""The solar system — the project's namesake, configurable end to end.

One renderer (`warp_shaders.cosmos`) draws any mix of stars (sun / neutron star
/ white dwarf / black hole) and configurable planets on chosen orbits, plus an
optional nebula. These scenes expose the named presets; ``--frames`` animates
the orbits.

    python render.py --scene solar_system --frames 60 --fps 12 --gif out/ss.gif
    python render.py --scene ss_blackhole --time 0 -o bh_system.png
"""

from ..cosmos import presets
from ..cosmos.system import render_system
from ..scene import Scene


def _scene(preset_name):
    def _render(width, height, time, mouse, device):
        syscfg = presets.get(preset_name)
        return render_system(syscfg, width, height, time=time, device=device)
    return _render


SCENES = [
    Scene(name="solar_system", renderer=_scene("first"),
          description="The first solar system: a live precessing neutron star + "
                      "one planet on an inclined elliptical orbit. --frames to "
                      "orbit."),
    Scene(name="ss_binary", renderer=_scene("binary"),
          description="Solar system: two suns (a binary) with an earth-like "
                      "planet orbiting the pair."),
    Scene(name="ss_trinary", renderer=_scene("trinary"),
          description="Solar system: three stars (sun / neutron star / white "
                      "dwarf) with a gas-giant planet."),
    Scene(name="ss_blackhole", renderer=_scene("blackhole"),
          description="Solar system: a black hole lensing a companion sun and a "
                      "planet."),
    Scene(name="ss_nebula", renderer=_scene("nebula_cradle"),
          description="Solar system: a sun with two planets, cradled in a "
                      "positioned nebula."),
]
