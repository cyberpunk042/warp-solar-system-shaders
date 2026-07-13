"""A lightning storm — bolts forking out of dark thunderclouds.

Turbulent fBm storm clouds, lit from within by branching **lightning** bolts that
strike on a timer — a jagged main channel with forks, flashing white-blue and
illuminating the cloud, with rain below. See
``docs/research/25-earth-and-weather.md``. --frames animates the strikes.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.noise import fbm3
from ..procedural.hash import hash21
from ..scene import Scene


def hash21_np(n):
    return float(np.modf(np.sin(n * 12.9898) * 43758.5453)[0])


@wp.func
def _bolt_x(y: float, seed: float, wobble: float) -> float:
    # a jagged near-vertical channel descending with y (top→bottom)
    return seed + wobble * (0.22 * wp.sin(y * 6.0 + seed * 12.0)
                            + 0.1 * wp.sin(y * 17.0 + seed * 5.0)
                            + 0.05 * wp.sin(y * 41.0))


@wp.kernel
def storm_kernel(img: wp.array2d(dtype=wp.vec3), seed: float, flash: float,
                 nfork: int, aspect: float, time: float, width: int, height: int):
    i, j = wp.tid()
    x = (((float(j) + 0.5) / float(width)) - 0.5) * 2.0 * aspect
    y = ((float(height - 1 - i) + 0.5) / float(height) - 0.5) * 2.0

    # storm clouds in the upper sky, rain haze below
    cl = fbm3(wp.vec3(x * 1.4, y * 1.8 - time * 0.05, 3.0), 6)
    cloud = wp.smoothstep(0.35, 0.75, cl) * wp.smoothstep(-0.6, 0.4, y)
    sky = wp.vec3(0.02, 0.03, 0.06)
    cloudc = wp.vec3(0.1, 0.11, 0.15) * (0.6 + 0.8 * cl)
    col = sky * (1.0 - cloud) + cloudc * cloud
    # rain streaks below the clouds
    rain = fbm3(wp.vec3(x * 30.0 + y * 8.0, y * 3.0 - time * 6.0, 1.0), 2)
    col = col + wp.vec3(0.1, 0.12, 0.16) * (wp.smoothstep(0.6, 0.9, rain) * wp.smoothstep(0.3, -0.8, y) * 0.4)

    # the lightning — main channel + forks, only during a flash
    if flash > 0.01:
        bx = _bolt_x(y, seed, 1.0)
        d = wp.abs(x - bx)
        top = wp.smoothstep(1.0, 0.9, y)              # emerges from the cloud top
        bolt = wp.exp(-(d / 0.006) * (d / 0.006)) + 0.4 * wp.exp(-(d / 0.03) * (d / 0.03))
        # forks branching off lower down
        for k in range(nfork):
            fk = float(k + 1)
            ys = 0.4 - 0.3 * fk                        # fork start height
            if y < ys:
                fx = _bolt_x(ys, seed, 1.0) + (ys - y) * (0.5 * (hash21(wp.vec2(seed, fk)) - 0.5)) * 2.0 \
                    + 0.08 * wp.sin(y * 30.0 + fk)
                df = wp.abs(x - fx)
                bolt = bolt + (wp.exp(-(df / 0.005) * (df / 0.005)) + 0.3 * wp.exp(-(df / 0.025) * (df / 0.025))) * 0.7
        bolt = bolt * top
        col = col + wp.vec3(0.7, 0.8, 1.0) * (bolt * flash * 2.2)
        # the flash lights the whole cloud
        col = col + wp.vec3(0.5, 0.55, 0.7) * (cloud * flash * 0.5)
    img[i, j] = col


def _render(width, height, time, mouse, device, period=2.3):
    strike = float(math.floor(time / period))
    prog = (time % period) / period
    seed = (hash21_np(strike) - 0.5) * 1.4              # a new bolt each strike
    flash = 0.0
    if prog < 0.16:                                     # brief flash + flicker
        flash = (1.0 - prog / 0.16)
        flash = flash * (0.6 + 0.4 * np.sin(prog * 90.0))
        flash = max(flash, 0.0)
    nfork = 3
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(storm_kernel, dim=(height, width),
              inputs=[img, float(seed), float(flash), int(nfork),
                      float(width / height), float(time), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(3, int(min(width, height) * 0.02))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.55, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="lightning_storm",
    description="A lightning storm — branching bolts forking out of dark fBm "
                "thunderclouds and lighting them from within, with rain below. "
                "--frames animates the strikes.",
    renderer=_render,
)
