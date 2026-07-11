"""A living cell dividing — mitosis, in the glow-impostor style.

The smallest complete life on the ladder: a membrane, cytoplasm, a nucleus and
organelles (:mod:`warp_shaders.life.cell`). As ``time`` advances the cell
**divides** — the nucleus parts, organelles partition, the membrane pinches into
two daughters.

    python render.py --scene cell --frames 24 --fps 12 --gif out/cell.gif
    python render.py --scene cell --time 6 -o cell.png
"""

from ..life.cell import render_cell
from ..scene import Scene


def _smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _render(width, height, time, mouse, device):
    # rest as one cell for ~1s, then divide over the next ~4s
    divide = _smooth((time - 1.0) / 4.0)
    return render_cell(width, height, time, mouse, divide, device)


SCENE = Scene(name="cell", renderer=_render,
              description="A cell dividing (mitosis): membrane pinches, nucleus "
                          "and organelles partition into two. --time 0..6.")
