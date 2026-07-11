"""The solar system — the project's namesake, configurable end to end.

One renderer (`warp_shaders.cosmos`) draws any mix of stars (sun / neutron star
/ white dwarf / black hole) and configurable planets on chosen orbits, plus an
optional nebula. These scenes expose the named presets; ``--frames`` animates
the orbits.

    python render.py --scene solar_system --frames 60 --fps 12 --gif out/ss.gif
    python render.py --scene ss_blackhole --time 0 -o bh_system.png
"""

import math

from ..cosmos import presets
from ..cosmos.system import render_system
from ..engine.camera_path import fly
from ..scene import Scene

_LOOP = 12.0                                # seconds for one full camera orbit


def _scene(preset_name):
    def _render(width, height, time, mouse, device):
        syscfg = presets.get(preset_name)
        return render_system(syscfg, width, height, time=time, device=device)
    return _render


def _orbit_eye(r, el, az):
    return (r * math.cos(el) * math.sin(az), r * math.sin(el),
            r * math.cos(el) * math.cos(az))


def _flyby(preset_name):
    """A cinematic camera sweep around a system: orbit once per `_LOOP` seconds
    while easing the elevation and gently pushing in, driven by a Catmull-Rom
    path (loops seamlessly — first and last keyframes coincide)."""
    def _render(width, height, time, mouse, device):
        sys = presets.get(preset_name)
        r, f = sys.dist, sys.fov
        pi = math.pi
        path = fly([
            (0.00, _orbit_eye(r * 1.15, 0.10, 0.0),        (0, 0, 0), f + 4.0),
            (0.25, _orbit_eye(r * 1.00, 0.38, 0.5 * pi),   (0, 0, 0), f),
            (0.50, _orbit_eye(r * 0.88, 0.14, pi),         (0, 0, 0), f - 3.0),
            (0.75, _orbit_eye(r * 1.00, 0.46, 1.5 * pi),   (0, 0, 0), f),
            (1.00, _orbit_eye(r * 1.15, 0.10, 2.0 * pi),   (0, 0, 0), f + 4.0),
        ], easing="ease_in_out")
        pt = (time / _LOOP) % 1.0
        cam = path.sample(pt)
        return render_system(sys, width, height, time=time, device=device, camera=cam)
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
    Scene(name="ss_flyby", renderer=_flyby("trinary"),
          description="Cinematic camera fly-by of the trinary system — a keyframed "
                      "Catmull-Rom orbit (elevation ease + push-in), loops every "
                      "12s. Use --frames/--video."),
]
