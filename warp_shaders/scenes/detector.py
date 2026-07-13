"""Detector scenes — the bubble chamber and a collider collision event."""

from ..scene import Scene
from ..subatomic.detector import render_bubble_chamber, render_collision

SCENES = [
    Scene(name="bubble_chamber",
          description="a bubble chamber — charged particles curling into curved "
                      "tracks in a magnetic field (curvature ∝ 1/momentum, opposite "
                      "charges bend opposite ways), radiating from a vertex with a "
                      "neutral-decay V. --frames drifts the field.",
          renderer=render_bubble_chamber),
    Scene(name="particle_collision",
          description="a collider event display — two beams meet and spray a fan of "
                      "curved tracks in every direction from the collision vertex, "
                      "flashing at the centre. iMouse orbits.",
          renderer=render_collision),
]
