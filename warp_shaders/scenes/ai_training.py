"""AI training — gradient descent on a loss landscape.

A neural network is a mesh of weighted connections whose weights are tuned by
**gradient descent** to minimise error. Picture the error as a **loss landscape**:
a surface of hills and valleys over the weight space, and a ball that **rolls
downhill** — following the negative gradient — until it settles into a minimum.
That settling *is* learning. See ``docs/research/26-the-machine.md``. --frames
rolls the ball down.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..scene import Scene

# loss-landscape wells: (mx, mz, depth, width). First is the wide global minimum.
_WELLS = [
    (0.15, -0.05, 0.95, 0.75),
    (-1.15, 0.85, 0.40, 0.34),
    (1.15, 0.75, 0.42, 0.36),
    (-0.9, -1.05, 0.34, 0.32),
    (1.0, -1.1, 0.30, 0.30),
]


def _loss_np(x, z):
    h = 1.0 + 0.05 * (x * x + z * z)
    for mx, mz, dp, w in _WELLS:
        h -= dp * math.exp(-((x - mx) ** 2 + (z - mz) ** 2) / (2.0 * w * w))
    return h


def _grad_np(x, z, e=1e-3):
    return ((_loss_np(x + e, z) - _loss_np(x - e, z)) / (2 * e),
            (_loss_np(x, z + e) - _loss_np(x, z - e)) / (2 * e))


def _trajectory(x0, z0, steps=600, lr=0.016, mom=0.72):
    xs = []
    x, z, vx, vz = x0, z0, 0.0, 0.0
    for _ in range(steps):
        gx, gz = _grad_np(x, z)
        vx = mom * vx - lr * gx
        vz = mom * vz - lr * gz
        x += vx
        z += vz
        xs.append((x, z))
    return xs


_TRAJ = _trajectory(1.55, 1.5)


@wp.func
def _well(x: float, z: float, mx: float, mz: float, dp: float, w: float) -> float:
    return dp * wp.exp(-((x - mx) * (x - mx) + (z - mz) * (z - mz)) / (2.0 * w * w))


@wp.func
def _loss(x: float, z: float) -> float:
    h = 1.0 + 0.05 * (x * x + z * z)
    h = h - _well(x, z, 0.15, -0.05, 0.95, 0.75)
    h = h - _well(x, z, -1.15, 0.85, 0.40, 0.34)
    h = h - _well(x, z, 1.15, 0.75, 0.42, 0.36)
    h = h - _well(x, z, -0.9, -1.05, 0.34, 0.32)
    h = h - _well(x, z, 1.0, -1.1, 0.30, 0.30)
    return h


@wp.kernel
def loss_kernel(img: wp.array2d(dtype=wp.vec3), ball: wp.vec3, aspect: float,
                width: int, height: int):
    i, j = wp.tid()
    u = (((float(j) + 0.5) / float(width)) * 2.0 - 1.0) * aspect
    v = ((float(height - 1 - i) + 0.5) / float(height)) * 2.0 - 1.0

    eye = wp.vec3(0.0, 2.35, 3.35)
    tgt = wp.vec3(0.0, 0.35, 0.0)
    fwd = wp.normalize(tgt - eye)
    rgt = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    upv = wp.cross(rgt, fwd)
    rd = wp.normalize(fwd * 1.6 + rgt * u + upv * v)

    col = wp.vec3(0.02, 0.03, 0.05) + wp.vec3(0.01, 0.02, 0.05) * (0.5 + 0.5 * rd[1])
    # heightfield march (surface y = loss(x,z), rendered below y0)
    t = float(0.1)
    hit = float(-1.0)
    hx = float(0.0)
    hz = float(0.0)
    for _ in range(140):
        p = eye + rd * t
        h = _loss(p[0], p[2]) * 0.95             # surface height = loss (valleys low)
        if p[1] < h and p[0] > -2.2 and p[0] < 2.2 and p[2] > -2.2 and p[2] < 2.2:
            hit = t
            hx = p[0]
            hz = p[2]
            break
        t += wp.max(0.02, (p[1] - h) * 0.4)
        if t > 9.0:
            break

    # the ball (gradient-descent optimiser) — sphere test
    br = 0.11
    oc = eye - ball
    bb = wp.dot(oc, rd)
    cc = wp.dot(oc, oc) - br * br
    disc = bb * bb - cc
    tb = float(-1.0)
    if disc > 0.0:
        tb = -bb - wp.sqrt(disc)

    if tb > 0.0 and (hit < 0.0 or tb < hit):
        p = eye + rd * tb
        nrm = wp.normalize(p - ball)
        lite = wp.max(wp.dot(nrm, wp.normalize(wp.vec3(0.4, 0.8, 0.5))), 0.0)
        col = wp.vec3(1.0, 0.75, 0.2) * (0.3 + 0.9 * lite) + wp.vec3(0.4, 0.2, 0.05)
        col = col + wp.vec3(1.0, 0.9, 0.6) * wp.pow(lite, 24.0)
    elif hit > 0.0:
        loss = _loss(hx, hz)
        lo = wp.clamp((loss - 0.15) / 0.95, 0.0, 1.0)        # 0 good .. 1 bad
        cool = wp.vec3(0.1, 0.7, 0.65)
        warm = wp.vec3(0.85, 0.2, 0.55)
        base = cool * (1.0 - lo) + warm * lo
        # contour grid on the surface
        gx = wp.abs(wp.mod(hx * 2.0 + 100.0, 1.0) - 0.5)
        gz = wp.abs(wp.mod(hz * 2.0 + 100.0, 1.0) - 0.5)
        grid = wp.max(wp.exp(-gx * gx * 260.0), wp.exp(-gz * gz * 260.0))
        col = base * (0.5 + 0.35 * (1.0 - lo)) + wp.vec3(0.6, 0.9, 1.0) * grid * 0.28

    img[i, j] = col


def _render(width, height, time, mouse, device):
    idx = min(int(time * 26.0), len(_TRAJ) - 1)
    bx, bz = _TRAJ[idx]
    by = _loss_np(bx, bz) * 0.95 + 0.11
    ball = wp.vec3(float(bx), float(by), float(bz))
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(loss_kernel, dim=(height, width),
              inputs=[img, ball, float(width / height), int(width), int(height)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    r = max(2, int(min(width, height) * 0.01))
    hdr = post.bloom(hdr, threshold=1.1, strength=0.3, radius=r, passes=3)
    return post.tonemap(hdr, mode="aces", exposure=1.08)


SCENE = Scene(
    name="ai_training",
    description="Gradient descent on a loss landscape — a surface of hills and valleys "
                "over weight space (cool = low loss, warm = high) with a ball rolling "
                "downhill along the negative gradient until it settles in a minimum. "
                "That settling is learning. --frames rolls the ball down.",
    renderer=_render,
)
