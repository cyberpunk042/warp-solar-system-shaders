"""Data flow — a memory array and its bitstream.

All of it is **binary**: memory is a vast grid of **cells**, each holding a bit,
addressed by row and column. A **read** sweeps row by row and the bits stream out
along the **data bus** as a serial pulse train — the machine is patterns of
electricity moving through the switch-lattice. See ``docs/research/26-the-machine.md``.
--frames flips the cells and drives the readout.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.hash import hash21
from ..scene import Scene

_COLS = 28.0
_ROWS = 12.0


@wp.func
def _cell_bit(gx: float, gy: float, tick: float) -> float:
    # a bit that flips as the pattern updates each tick
    return wp.step(0.5 - hash21(wp.vec2(gx * 1.7 + 4.0, gy * 2.3 + tick * 13.0)))


@wp.kernel
def mem_kernel(img: wp.array2d(dtype=wp.vec3), aspect: float, tick: float,
              rrow: float, time: float, width: int, height: int):
    i, j = wp.tid()
    u = (float(j) + 0.5) / float(width)
    v = (float(height - 1 - i) + 0.5) / float(height)
    col = wp.vec3(0.015, 0.02, 0.03)

    if v > 0.24:                                    # memory array
        gv = (v - 0.24) / 0.76
        gx = wp.floor(u * _COLS)
        gy = wp.floor(gv * _ROWS)
        fx = u * _COLS - gx - 0.5
        fy = gv * _ROWS - gy - 0.5
        bit = _cell_bit(gx, gy, tick)
        # cell body (rounded square)
        d = wp.max(wp.abs(fx), wp.abs(fy))
        cell = wp.smoothstep(0.42, 0.30, d)
        oncol = wp.vec3(0.3, 0.95, 1.0)
        offcol = wp.vec3(0.05, 0.12, 0.18)
        base = oncol * bit + offcol * (1.0 - bit)
        col = col + base * cell * (0.5 + 1.2 * bit)
        # active read row: a bright address line + halo
        onrow = wp.exp(-((gy - rrow) * (gy - rrow)) / 0.4)
        col = col + wp.vec3(1.0, 0.85, 0.4) * cell * onrow * (0.4 + bit * 1.2)
        # thin column bus lines between cells
        colline = wp.exp(-(fx * fx) * 300.0)
        col = col + wp.vec3(0.1, 0.25, 0.35) * colline * 0.25
    else:                                           # data bus (serial readout)
        # bit currently streaming = the read row's cells serialized
        busy = wp.exp(-((v - 0.12) * (v - 0.12)) / 0.0016)
        col = col + wp.vec3(0.12, 0.3, 0.4) * busy * 0.6
        # pulse train: bright dashes moving right, gated by a bit pattern
        s = u * _COLS - time * 5.0
        celln = wp.floor(s)
        b = wp.step(0.5 - hash21(wp.vec2(celln * 1.7 + 4.0, rrow * 2.3 + tick * 13.0)))
        ph = s - celln
        pul = wp.pow(wp.max(1.0 - wp.abs(ph - 0.5) * 3.0, 0.0), 2.0)
        col = col + wp.vec3(0.8, 1.0, 1.0) * busy * pul * b * 2.6

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tick = float(int(time * 1.4))
    rrow = float(int(time * 2.2) % int(_ROWS))
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(mem_kernel, dim=(height, width),
              inputs=[img, float(width / height), tick, rrow, float(time),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.0, strength=0.4, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.04)


SCENE = Scene(
    name="data_flow",
    description="A memory array and its bitstream — a grid of cells each holding a bit "
                "(lit = 1), a read sweeping row by row (amber address line), and the "
                "bits streaming out along the data bus as a serial pulse train. "
                "--frames flips the cells and drives the readout.",
    renderer=_render,
)
