"""Ion and positronium — charged and antimatter atoms."""

from ..scene import Scene
from ..subatomic.exotic import render_ion, render_positronium

SCENES = [
    Scene(name="ion",
          description="an ion (a cation) — an atom caught mid-ionisation: the nucleus "
                      "and a depleted electron cloud, the ejected electron streaking "
                      "away, and a warm net-positive charge halo. --frames ejects it.",
          renderer=render_ion),
    Scene(name="positronium",
          description="positronium — a hydrogen-like atom of an electron + positron "
                      "orbiting their common centre inside a shared cloud; a matter/"
                      "antimatter atom that annihilates in ~0.1 ns. iMouse orbits.",
          renderer=render_positronium),
]
