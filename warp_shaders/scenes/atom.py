"""Hydrogen atom — the electron's 1s probability cloud around a proton nucleus.

The electron is not a dot: bound to the proton it is a standing probability wave.
Here it is the true 1s density |ψ_{100}|² ∝ e^{−2r/a₀}, ray-marched as a
volumetric cloud around a small bright nucleus. NOT to scale (a real nucleus is
~1e-5 of the atom). See ``docs/research/21-standard-model.md``. iMouse orbits.
"""

from ..scene import Scene
from ..subatomic.atom import render_named


def _render(width, height, time, mouse, device):
    return render_named(width, height, time, mouse, device, orb=0, nucleus=1.6)


SCENE = Scene(
    name="atom",
    description="Hydrogen — the electron's real 1s probability cloud "
                "|ψ|² ∝ e^(−2r/a₀) ray-marched around a bright proton nucleus "
                "(not to scale). iMouse orbits.",
    renderer=_render,
)
