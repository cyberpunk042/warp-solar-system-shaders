"""A collapsing solar system — the destructive scenario, cached and played back.

Two massive suns spiral in under mutual gravity, merge, collapse (supernova
flash) into a black hole by mass, which then swallows the planet. The whole
N-body sequence is simulated once and cached; ``time`` maps to a frame.

    python render.py --scene ss_collapse --frames 60 --fps 16 --gif out/collapse.gif
    python render.py --scene ss_collapse --time 1.6 -o collapse.png
"""

import numpy as np

from ..cosmos import presets
from ..cosmos.dynamics import simulate
from ..scene import Scene

_DT = 0.05
_FRAMES = 62
_CW, _CH = 512, 320
_CACHE = {}


def _frames(device):
    key = ("collapse", device)
    if key not in _CACHE:
        _CACHE[key] = simulate(presets.get("collapse"), frames=_FRAMES, dt=_DT,
                               width=_CW, height=_CH, decay=0.26, device=device)
    return _CACHE[key]


def _render(width, height, time, mouse, device):
    frames = _frames(device)
    idx = max(0, min(_FRAMES - 1, int(round(time / _DT))))
    img = frames[idx]
    if (img.shape[1], img.shape[0]) != (width, height):
        ys = np.linspace(0, img.shape[0] - 1, height).astype(int)
        xs = np.linspace(0, img.shape[1] - 1, width).astype(int)
        img = img[ys][:, xs]
    return img


SCENE = Scene(name="ss_collapse", renderer=_render,
              description="A collapsing solar system: two suns spiral in, merge, "
                          "collapse to a black hole (supernova), which swallows "
                          "the planet. --frames 60 --fps 16.")
