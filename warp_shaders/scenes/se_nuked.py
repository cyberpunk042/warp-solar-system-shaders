"""super-earth under nuclear bombardment — configurable strikes raining down.

The whole sequence (warhead count, distribution formula, delay, interval,
parallel detonations) is rendered once and cached; the scene maps ``time`` to a
frame, so ``--frames``/``--gif`` plays the bombardment. Each strike flashes,
throws a fireball, and leaves an expanding shock-ring scar
(:mod:`warp_shaders.superearth.bombardment`).

    python render.py --scene se_nuked --frames 60 --fps 16 --gif out/nuked.gif
    python render.py --scene se_nuked --time 2.5 -o nuked.png
"""

from ..scene import Scene
from ..superearth.bombardment import BombConfig, run
from ..superearth.planet import make_config

_DT = 0.07
_FRAMES = 56
_DIST = 3.4
_FOV = 42.0
_CW, _CH = 420, 320
_CACHE = {}


def _frames(device):
    key = ("nuked", device)
    if key not in _CACHE:
        # a green living world (no clouds, so the strikes read clearly + render fast)
        cfg = make_config(seed=1.0, mountain=0.6, sea_level=0.0, has_ocean=1,
                          has_rivers=1, snow=1.0, has_atmo=1, atmo=1.0, veg=0.9,
                          cloud=0.0)
        bcfg = BombConfig(n=40, delay=0.3, interval=0.28, parallel=4,
                          formula="clustered", yield_scale=1.0, seed=2)
        _CACHE[key] = run(cfg, bcfg, _CW, _CH, _FRAMES, _DT, device, _DIST, _FOV,
                          quality="low")
    return _CACHE[key]


def _render(width, height, time, mouse, device):
    frames = _frames(device)
    idx = max(0, min(_FRAMES - 1, int(round(time / _DT))))
    import numpy as np
    img = frames[idx]
    if (img.shape[1], img.shape[0]) != (width, height):
        # nearest-neighbour resize to the requested size (cache is 480x360)
        ys = (np.linspace(0, img.shape[0] - 1, height)).astype(int)
        xs = (np.linspace(0, img.shape[1] - 1, width)).astype(int)
        img = img[ys][:, xs]
    return img


SCENE = Scene(name="se_nuked", renderer=_render,
              description="super-earth under a configurable nuclear bombardment "
                          "(count / distribution / delay / interval / parallel). "
                          "--frames 60 --fps 16.")
