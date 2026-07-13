"""Rock salt (NaCl) — the face-centred-cubic ionic crystal.

A domain-repeated lattice of alternating **Na⁺** (violet, small) and **Cl⁻** (green,
large) ions on a cubic grid, with thin rods along the lattice lines, carved to a
cube. Each ion is octahedrally surrounded by six of the other. Sphere-traced with
studio lighting. See ``docs/research/22-chemistry-and-molecules.md``. iMouse orbits.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.uniforms import Camera, camera_ray_dir
from ..subatomic.render import orbit_camera
from ..scene import Scene

_L = wp.constant(2.5)                                # half-extent of the carved cube


@wp.func
def _parity(cell: wp.vec3) -> int:
    s = int(cell[0]) + int(cell[1]) + int(cell[2])
    return s - 2 * (s / 2)                            # 0 = Na, 1 = Cl (integer mod 2)


@wp.func
def _box(p: wp.vec3, b: float) -> float:
    q = wp.vec3(wp.abs(p[0]) - b, wp.abs(p[1]) - b, wp.abs(p[2]) - b)
    return wp.length(wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))) \
        + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0)


@wp.func
def _lattice(p: wp.vec3) -> float:
    cell = wp.vec3(wp.floor(p[0] + 0.5), wp.floor(p[1] + 0.5), wp.floor(p[2] + 0.5))
    local = p - cell
    r = 0.33
    if _parity(cell) == 1:
        r = 0.42
    ds = wp.length(local) - r
    rx = wp.length(wp.vec2(local[1], local[2])) - 0.07
    ry = wp.length(wp.vec2(local[0], local[2])) - 0.07
    rz = wp.length(wp.vec2(local[0], local[1])) - 0.07
    return wp.min(ds, wp.min(rx, wp.min(ry, rz)))


@wp.func
def _map(p: wp.vec3) -> float:
    return wp.max(_lattice(p), _box(p, _L))


@wp.func
def _nrm(p: wp.vec3) -> wp.vec3:
    e = 0.012
    return wp.normalize(wp.vec3(
        _map(p + wp.vec3(e, 0.0, 0.0)) - _map(p - wp.vec3(e, 0.0, 0.0)),
        _map(p + wp.vec3(0.0, e, 0.0)) - _map(p - wp.vec3(0.0, e, 0.0)),
        _map(p + wp.vec3(0.0, 0.0, e)) - _map(p - wp.vec3(0.0, 0.0, e))))


@wp.kernel
def crystal_kernel(img: wp.array2d(dtype=wp.vec3), cam: Camera,
                   width: int, height: int):
    i, j = wp.tid()
    u = (2.0 * (float(j) + 0.5) / float(width)) - 1.0
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height)) - 1.0
    ro = cam.eye
    rd = camera_ray_dir(cam, u, v)
    bg = wp.vec3(0.03, 0.035, 0.05) + wp.vec3(0.05, 0.06, 0.08) * (0.5 + 0.5 * v)

    t = float(0.0)
    hit = int(0)
    for _ in range(120):
        p = ro + rd * t
        d = _map(p)
        if d < 0.001:
            hit = 1
            break
        t = t + d
        if t > 18.0:
            break
    col = bg
    if hit == 1:
        p = ro + rd * t
        n = _nrm(p)
        cell = wp.vec3(wp.floor(p[0] + 0.5), wp.floor(p[1] + 0.5), wp.floor(p[2] + 0.5))
        local = p - cell
        r = 0.33
        if _parity(cell) == 1:
            r = 0.42
        is_ion = wp.step(wp.length(local) - r - 0.02)     # <0.02 → on an ion
        na = wp.vec3(0.6, 0.35, 0.9)
        cl = wp.vec3(0.3, 0.9, 0.4)
        ion = na
        if _parity(cell) == 1:
            ion = cl
        base = wp.vec3(0.6, 0.62, 0.66) * (1.0 - is_ion) + ion * is_ion
        key = wp.normalize(wp.vec3(0.6, 0.8, 0.5))
        ndl = wp.max(wp.dot(n, key), 0.0)
        h = wp.normalize(key - rd)
        spec = wp.pow(wp.max(wp.dot(n, h), 0.0), 40.0)
        fres = wp.pow(1.0 - wp.max(wp.dot(n, -rd), 0.0), 3.0)
        col = base * (0.25 + 0.85 * ndl) + wp.vec3(1.0, 1.0, 1.0) * (spec * 0.5) + base * (fres * 0.4)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    cam = orbit_camera(width, height, time, mouse, dist=9.5, fov=40.0, el0=0.4)
    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(crystal_kernel, dim=(height, width),
              inputs=[img, cam, int(width), int(height)], device=device)
    wp.synchronize_device(device)
    hdr = img.numpy().astype(np.float32)
    hdr = post.bloom(hdr, threshold=1.5, strength=0.2, radius=3, passes=2)
    return post.tonemap(hdr, mode="aces", exposure=1.05)


SCENE = Scene(
    name="salt_crystal",
    description="Rock salt (NaCl) — a cubic lattice of alternating Na⁺ (violet) and "
                "Cl⁻ (green) ions with lattice rods, carved to a cube. iMouse orbits.",
    renderer=_render,
)
