"""Super Tsar — a hypothetical 500 Mt device (10× Tsar Bomba) over a forest.

Same physics as `tsar_bomba`, yield scaled ×10: by the scaling laws the fireball
grows ×10^0.4 ≈ 2.5 (to ~8.7 km) and every blast ring ×10^(1/3) ≈ 2.15 — a far
larger fireball and a wider swath of flattened, scorched forest. See
``docs/research/15-nuclear-fireball.md``.
"""

from ..blast import physics as P
from ..blast.render import render_ground
from ..scene import Scene


def _render(width, height, time, mouse, device):
    return render_ground(width, height, time, mouse, device, P.SUPER_TSAR_KT)


SCENE = Scene(
    name="super_tsar",
    description="Super Tsar (500 Mt, 10x Tsar Bomba) over a forest — a ~8.7 km "
                "fireball and a far wider flattened, scorched blast zone.",
    renderer=_render,
)
