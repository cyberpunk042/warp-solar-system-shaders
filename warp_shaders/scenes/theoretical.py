"""Theoretical particles — tachyon, graviton, magnetic monopole, axion, dark matter.

Particles that are predicted by theory but have never been observed. See
``docs/research/21-standard-model.md``.
"""

from ..scene import Scene
from ..subatomic.theoretical import render_graviton, render_monopole, render_tachyon
from ..subatomic.hypothetical import render_axion, render_dark_matter

SCENES = [
    Scene(name="tachyon",
          description="a tachyon — a hypothetical faster-than-light particle "
                      "(imaginary rest mass) dragging a Cherenkov shock cone of "
                      "blueshifted light. iMouse orbits.",
          renderer=render_tachyon),
    Scene(name="graviton",
          description="a graviton — the conjectured spin-2 quantum of gravity, a "
                      "ripple stretching and squeezing a spacetime grid by its "
                      "quadrupole (plus-polarisation) strain. --frames ripples it.",
          renderer=render_graviton),
    Scene(name="magnetic_monopole",
          description="a magnetic monopole — an isolated magnetic charge with radial "
                      "field lines streaming out in every direction (unlike a real "
                      "magnet, whose field always loops). iMouse orbits.",
          renderer=render_monopole),
    Scene(name="axion",
          description="an axion — an ultralight dark-matter candidate, a faint "
                      "pseudoscalar shimmer converting into photons in a magnetic "
                      "field (the Primakoff effect). iMouse orbits.",
          renderer=render_axion),
    Scene(name="dark_matter",
          description="a dark-matter particle (WIMP) — invisible and non-luminous, "
                      "revealed only by the gravitational lensing that bends the "
                      "background starlight around its mass. iMouse orbits.",
          renderer=render_dark_matter),
]
