"""Super Tsar in space — a 500 Mt vacuum burst above a planet.

With no atmosphere the physics changes completely: no medium to carry a blast
wave, no air to heat into an incandescent fireball, no buoyancy for a mushroom.
The energy leaves as an X-ray flash and the vaporised debris expands as a thin,
ballistic **plasma shell** (radius linear in time), thinning and cooling — the
1962 Starfish Prime behaviour. The planet and its gravity sit in frame, untouched
by any blast. See ``docs/research/15-nuclear-fireball.md``.
"""

from ..blast import physics as P
from ..blast.render import render_space
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_space(width, height, time, mouse, device, P.SUPER_TSAR_KT)


SCENE = Scene(
    name="super_tsar_space",
    description="Super Tsar (500 Mt) detonated in vacuum above a planet — no "
                "blast/fireball/mushroom, a ballistic plasma shell (Starfish Prime).",
    renderer=_render,
)
