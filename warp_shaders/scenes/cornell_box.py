"""Cornell box — real global illumination by Monte-Carlo path tracing.

The engine leap: instead of a single-bounce raymarch, this scene **path-traces** the
canonical Cornell box — rays bounce diffusely around the room, gathering light over many
samples, so the red and green walls **bleed** their colour onto the white boxes and floor,
soft contact shadows form under the blocks, and the whole thing is lit only by the small
emissive patch on the ceiling. Cosine-weighted hemisphere sampling, a few bounces, dozens
of samples per pixel, averaged. See ``docs/research/39-engine-leap.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.sdf import op_union, sd_box
from ..scene import Scene

_RX = 1.0
_RY = 1.0
_RZ = 1.0
_SPP = 384          # samples per pixel
_BOUNCES = 4


@wp.func
def _scene(p: wp.vec3) -> float:
    # five finite walls with an OPEN FRONT (+z), so the camera frames the whole box
    th = 0.04
    floor = sd_box(p - wp.vec3(0.0, -_RY, 0.0), wp.vec3(_RX, th, _RZ))
    ceil = sd_box(p - wp.vec3(0.0, _RY, 0.0), wp.vec3(_RX, th, _RZ))
    back = sd_box(p - wp.vec3(0.0, 0.0, -_RZ), wp.vec3(_RX, _RY, th))
    left = sd_box(p - wp.vec3(-_RX, 0.0, 0.0), wp.vec3(th, _RY, _RZ))
    right = sd_box(p - wp.vec3(_RX, 0.0, 0.0), wp.vec3(th, _RY, _RZ))
    walls = op_union(op_union(floor, ceil), op_union(back, op_union(left, right)))
    tall = sd_box(p - wp.vec3(-0.35, -0.4, -0.35), wp.vec3(0.3, 0.6, 0.3))
    short = sd_box(p - wp.vec3(0.4, -0.7, 0.35), wp.vec3(0.3, 0.3, 0.3))
    return op_union(walls, op_union(tall, short))


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0015
    dx = _scene(p + wp.vec3(e, 0.0, 0.0)) - _scene(p - wp.vec3(e, 0.0, 0.0))
    dy = _scene(p + wp.vec3(0.0, e, 0.0)) - _scene(p - wp.vec3(0.0, e, 0.0))
    dz = _scene(p + wp.vec3(0.0, 0.0, e)) - _scene(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _material(p: wp.vec3, alb: wp.vec3) -> wp.vec3:
    # colour a hit by which surface it belongs to (red left wall, green right wall,
    # white everything else). Returns an albedo.
    a = alb
    if p[0] < -0.9:
        a = wp.vec3(0.62, 0.06, 0.06)                      # left wall red
    elif p[0] > 0.9:
        a = wp.vec3(0.10, 0.48, 0.10)                      # right wall green
    else:
        a = wp.vec3(0.72, 0.72, 0.72)                      # white
    return a


@wp.func
def _emission(p: wp.vec3) -> wp.vec3:
    # the ceiling light patch (underside of the ceiling slab)
    if p[1] > 0.9 and wp.abs(p[0]) < 0.4 and wp.abs(p[2]) < 0.4:
        return wp.vec3(16.0, 13.0, 9.0)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _march(ro: wp.vec3, rd: wp.vec3):
    t = float(0.002)
    hit = int(0)
    for _ in range(96):
        p = ro + rd * t
        d = _scene(p)
        if d < 0.0006:
            hit = 1
            break
        t += wp.max(d * 0.9, 0.0008)
        if t > 12.0:
            break
    return t, hit


@wp.func
def _onb(n: wp.vec3, r1: float, r2: float) -> wp.vec3:
    # cosine-weighted hemisphere sample around n
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    tang = wp.normalize(wp.cross(a, n))
    bit = wp.cross(n, tang)
    r = wp.sqrt(r1)
    phi = 6.2831853 * r2
    x = r * wp.cos(phi)
    y = r * wp.sin(phi)
    z = wp.sqrt(wp.max(0.0, 1.0 - r1))
    return wp.normalize(tang * x + bit * y + n * z)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanfov: float,
                   spp: int, seed0: int):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    acc = wp.vec3(0.0, 0.0, 0.0)
    pix = i * width + j

    for s in range(spp):
        rng = wp.rand_init(seed0 + pix * 9781 + s * 6151, s * 2749 + 17)
        du = (wp.randf(rng) - 0.5)
        dv = (wp.randf(rng) - 0.5)
        u = (2.0 * (float(j) + 0.5 + du) / float(width) - 1.0) * tanfov * aspect
        v = (2.0 * (float(height - 1 - i) + 0.5 + dv) / float(height) - 1.0) * tanfov
        rd = wp.normalize(fwd + right * u + up * v)
        ro = eye
        throughput = wp.vec3(1.0, 1.0, 1.0)
        radiance = wp.vec3(0.0, 0.0, 0.0)

        for _b in range(_BOUNCES):
            t, hit = _march(ro, rd)
            if hit == 0:
                break
            p = ro + rd * t
            n = _normal(p)
            if wp.dot(n, rd) > 0.0:
                n = -n                                     # face the ray (room walls)
            emit = _emission(p)
            if emit[0] > 0.0:
                radiance = radiance + wp.cw_mul(throughput, emit)
                break
            alb = _material(p, wp.vec3(0.72, 0.72, 0.72))
            throughput = wp.cw_mul(throughput, alb)
            ro = p + n * 0.002
            rd = _onb(n, wp.randf(rng), wp.randf(rng))
        acc = acc + radiance

    img[i, j] = acc / float(spp)


def _render(width, height, time, mouse, device):
    spp = _SPP
    if width * height <= 96 * 72:
        spp = 8                                            # fast path for the smoke test

    # camera sits back from the open front, framing the whole box
    eye = wp.vec3(0.0 + float(mouse[0]) * 0.004, 0.0, 2.75)
    tgt = wp.vec3(0.0, 0.0, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(42.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov, spp, 12345],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    return post.tonemap(hdr, mode="aces", exposure=1.5, preserve_hue=True)


SCENE = Scene(
    name="cornell_box",
    description="the classic Cornell box, path-traced — diffuse rays bouncing around the "
                "room so the red and green walls bleed colour onto the white blocks and "
                "floor, soft contact shadows, lit only by a ceiling patch. Real global "
                "illumination (Monte-Carlo path tracing).",
    renderer=_render,
)
