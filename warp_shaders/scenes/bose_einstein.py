"""A Bose–Einstein condensate — matter collapsing into one quantum state.

Cool bosonic atoms toward absolute zero and, below a critical temperature, they all
collapse into the *same* quantum state — one giant matter-wave. The iconic signature:
a **sharp central peak** rising out of the broad thermal cloud in the velocity
distribution as the gas is cooled (Cornell, Wieman, Ketterle, 1995). Rendered as that
distribution surface in the classic colour map. See ``docs/research/31-states-of-matter.md``.
--frames cools the gas.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene


@wp.func
def _bec(x: float, z: float, ath: float, ac: float) -> float:
    r2 = x * x + z * z
    return ath * wp.exp(-r2 / 1.1) + ac * wp.exp(-r2 / 0.045)


@wp.func
def _jet(t: float) -> wp.vec3:
    tt = wp.clamp(t, 0.0, 1.0)
    r = wp.clamp(1.5 - wp.abs(4.0 * tt - 3.0), 0.0, 1.0)
    g = wp.clamp(1.5 - wp.abs(4.0 * tt - 2.0), 0.0, 1.0)
    b = wp.clamp(1.5 - wp.abs(4.0 * tt - 1.0), 0.0, 1.0)
    col = wp.vec3(r, g, b)
    return col + wp.vec3(1.0, 1.0, 1.0) * wp.smoothstep(0.85, 1.0, tt)   # white peak


@wp.kernel
def bec_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
               rgt: wp.vec3, upv: wp.vec3, aspect: float, ath: float, ac: float,
               width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    rd = wp.normalize(fwd * 1.6 + rgt * (u * aspect) + upv * v)

    col = wp.vec3(0.02, 0.02, 0.035) + wp.vec3(0.01, 0.02, 0.04) * (0.5 + 0.5 * rd[1])
    t = float(0.1)
    for _ in range(150):
        p = eye + rd * t
        h = _bec(p[0], p[2], ath, ac)
        if p[1] < h and p[0] > -2.6 and p[0] < 2.6 and p[2] > -2.6 and p[2] < 2.6:
            e = 0.012
            n = wp.normalize(wp.vec3(
                _bec(p[0] - e, p[2], ath, ac) - _bec(p[0] + e, p[2], ath, ac),
                2.0 * e,
                _bec(p[0], p[2] - e, ath, ac) - _bec(p[0], p[2] + e, ath, ac)))
            base = _jet(h / 1.45)
            shade = 0.55 + 0.5 * wp.max(n[1], 0.0)
            gx = wp.abs(wp.mod(p[0] * 2.0 + 50.0, 1.0) - 0.5)
            gz = wp.abs(wp.mod(p[2] * 2.0 + 50.0, 1.0) - 0.5)
            grid = wp.max(wp.exp(-gx * gx * 300.0), wp.exp(-gz * gz * 300.0))
            col = base * shade + wp.vec3(0.1, 0.1, 0.12) * grid * 0.3
            break
        t += wp.max((p[1] - h) * 0.4, 0.02)
        if t > 12.0:
            break

    img[i, j] = col


def _render(width, height, time, mouse, device):
    cool = 0.5 + 0.5 * float(np.sin(time * 0.5))       # cooling cycle
    ath = 0.34 * (1.0 - 0.55 * cool)
    ac = 1.35 * cool
    eye = wp.vec3(0.0, 1.9, 2.9)
    tgt = wp.vec3(0.0, 0.35, 0.0)
    fwd = wp.normalize(tgt - eye)
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(bec_kernel, dim=(height, width),
              inputs=[img, eye, fwd, rgt, upv, float(width / height), float(ath), float(ac),
                      int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.008))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="bose_einstein",
    description="A Bose–Einstein condensate — the atom velocity distribution as the gas "
                "is cooled: a sharp central peak (all atoms in one quantum state) rising "
                "out of the broad thermal cloud, in the classic BEC colour map. "
                "--frames cools the gas.",
    renderer=_render,
)
