"""Tsar Bomba — the 50 Mt detonation over a forested landscape.

A physically-sized nuclear fireball (blackbody, ~3.5 km radius), an expanding
condensation shock ring flattening + charring the forest inside the ~38 km
destruction ring, and the rising mushroom cloud — all sized from
`blast.physics` (see ``docs/research/15-nuclear-fireball.md``). Animate with
``--frames`` to watch the flash, the shock sweep, and the cloud climb.
`--quality` scales the march / volume steps.
"""

from ..blast import physics as P
from ..blast.render import render_ground
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_ground(width, height, time, mouse, device, P.TSAR_KT)


SCENE = Scene(
    name="tsar_bomba",
    description="Tsar Bomba (50 Mt) over a forest — physically-sized fireball, "
                "shock ring flattening trees, rising mushroom cloud.",
    renderer=_render,
)
