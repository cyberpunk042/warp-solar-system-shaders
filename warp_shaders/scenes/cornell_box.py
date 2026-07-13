"""Cornell box — real global illumination by Monte-Carlo path tracing.

The engine leap: instead of a single-bounce raymarch, this scene **path-traces** the
canonical Cornell box — rays bounce diffusely around the room, gathering light over many
samples, so the red and green walls **bleed** their colour onto the white boxes and floor,
soft contact shadows form under the blocks, and the whole thing is lit only by the small
emissive patch on the ceiling. Cosine-weighted hemisphere sampling, a few bounces, dozens
of samples per pixel, averaged. See ``docs/research/39-engine-leap.md``.
"""

import numpy as np
import warp as wp

from ..engine import post
from ..engine.pathtrace import camera_basis, onb_cosine, tanfov
from ..procedural.sdf import op_union, sd_box
from ..scene import Scene

_RX = 1.0
_RY = 1.0
_RZ = 1.0
_SPP = 160          # samples per pixel (next-event estimation → far less noise per sample)
_BOUNCES = 4

# the ceiling light, sampled directly for next-event estimation
_LY = wp.constant(0.955)                       # underside of the ceiling slab
_LH = wp.constant(0.4)                         # half-extent (x,z) of the emissive patch
_LAREA = wp.constant(0.64)                     # 0.8 × 0.8
_LEMIT = wp.constant(wp.vec3(16.0, 13.0, 9.0))
_INV_PI = wp.constant(0.31830989)


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
def _visible(a: wp.vec3, b: wp.vec3) -> float:
    # unshadowed test: march from a toward b, occluded if we hit anything before b
    d = b - a
    dist = wp.length(d)
    rd = d / dist
    t = float(0.004)
    while t < dist - 0.01:
        dd = _scene(a + rd * t)
        if dd < 0.0007:
            return 0.0
        t += wp.max(dd * 0.9, 0.0009)
    return 1.0


@wp.func
def _direct(p: wp.vec3, n: wp.vec3, alb: wp.vec3, r1: float, r2: float) -> wp.vec3:
    # next-event estimation: sample a point on the ceiling light and connect
    q = wp.vec3((r1 * 2.0 - 1.0) * _LH, _LY, (r2 * 2.0 - 1.0) * _LH)
    to = q - p
    dist = wp.length(to)
    wi = to / dist
    cos_s = wp.dot(n, wi)                       # cosine at the shaded surface
    cos_l = wi[1]                               # cosine at the light (normal points down: (0,-1,0))
    if cos_s <= 0.0 or cos_l <= 0.0:
        return wp.vec3(0.0, 0.0, 0.0)
    if _visible(p + n * 0.002, q) < 0.5:
        return wp.vec3(0.0, 0.0, 0.0)
    g = cos_s * cos_l / (dist * dist)
    # Lambertian BRDF = albedo/π; area-light estimator multiplies by the patch area
    return wp.cw_mul(alb * _INV_PI, _LEMIT) * (g * _LAREA)


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
        see_emit = float(1.0)                              # 1 only on the camera ray (NEE handles the rest)

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
                radiance = radiance + wp.cw_mul(throughput, emit) * see_emit
                break                                      # the light is a sink
            alb = _material(p, wp.vec3(0.72, 0.72, 0.72))
            radiance = radiance + wp.cw_mul(throughput,
                                            _direct(p, n, alb, wp.randf(rng), wp.randf(rng)))
            throughput = wp.cw_mul(throughput, alb)
            ro = p + n * 0.002
            rd = onb_cosine(n, wp.randf(rng), wp.randf(rng))
            see_emit = 0.0                                 # indirect emitter hits already counted by NEE
        acc = acc + radiance

    img[i, j] = acc / float(spp)


def _render(width, height, time, mouse, device):
    spp = _SPP
    if width * height <= 96 * 72:
        spp = 8                                            # fast path for the smoke test

    # camera sits back from the open front, framing the whole box
    eye = wp.vec3(0.0 + float(mouse[0]) * 0.004, 0.0, 2.75)
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(42.0), spp, 12345],
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
