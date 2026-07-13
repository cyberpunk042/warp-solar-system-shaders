"""Motion blur — a moving world, sampled across the shutter.

The engine leap continued. A single instant frozen perfectly is *less* real than a photograph:
a real camera's shutter is open for a slice of time, so anything that moves paints a smear on
the film. A path tracer gets this **for free** — the same Monte-Carlo machinery that jitters
rays in space also jitters them in **time**. Each sample picks a random instant within the
shutter, the geometry is evaluated *at that instant*, and averaging the samples smears whatever
moved while leaving static geometry razor-sharp.

Here three spheres translate left-to-right at increasing speeds (a sharp reference, then a
gentle smear, then a long streak) and a striped sphere **spins** so its bands blur into a soft
gradient — linear and rotational motion blur from one distributed integrator. See
``docs/research/39-engine-leap.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.sdf import op_union, sd_box, sd_sphere
from ..engine.pathtrace import onb_cosine
from ..scene import Scene

_SPP = 288
_BOUNCES = 3
_R = 0.3

# sphere lanes: base x, speed (units swept across the shutter), z, y
_Y = -0.35
_STATIC_X = wp.constant(-1.15)
_SLOW_X = wp.constant(-0.5)
_FAST_X = wp.constant(0.3)
_SPIN = wp.constant(wp.vec3(1.45, -0.35, -0.2))


@wp.func
def _mx(base: float, speed: float, tt: float, time: float) -> float:
    # centre x of a translating sphere at intra-shutter instant tt in [0,1]
    return base + speed * (tt - 0.5) + 0.15 * wp.sin(time * 6.2831853)


@wp.func
def _scene(p: wp.vec3, tt: float, time: float) -> float:
    floor = sd_box(p - wp.vec3(0.0, -0.7, 0.0), wp.vec3(2.6, 0.04, 1.6))
    back = sd_box(p - wp.vec3(0.0, 0.4, -1.2), wp.vec3(2.6, 1.4, 0.04))
    room = op_union(floor, back)
    s0 = sd_sphere(p - wp.vec3(_mx(_STATIC_X, 0.0, tt, time), _Y, 0.0), _R)     # sharp
    s1 = sd_sphere(p - wp.vec3(_mx(_SLOW_X, 0.5, tt, time), _Y, 0.0), _R)       # gentle smear
    s2 = sd_sphere(p - wp.vec3(_mx(_FAST_X, 0.95, tt, time), _Y, 0.0), _R)      # long streak
    spin = sd_sphere(p - _SPIN, _R)
    return op_union(room, op_union(op_union(s0, s1), op_union(s2, spin)))


@wp.func
def _normal(p: wp.vec3, tt: float, time: float) -> wp.vec3:
    e = 0.0015
    dx = _scene(p + wp.vec3(e, 0.0, 0.0), tt, time) - _scene(p - wp.vec3(e, 0.0, 0.0), tt, time)
    dy = _scene(p + wp.vec3(0.0, e, 0.0), tt, time) - _scene(p - wp.vec3(0.0, e, 0.0), tt, time)
    dz = _scene(p + wp.vec3(0.0, 0.0, e), tt, time) - _scene(p - wp.vec3(0.0, 0.0, e), tt, time)
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _albedo(p: wp.vec3, tt: float, time: float) -> wp.vec3:
    # which object owns this point (closest sphere), then its colour
    x0 = _mx(_STATIC_X, 0.0, tt, time)
    x1 = _mx(_SLOW_X, 0.5, tt, time)
    x2 = _mx(_FAST_X, 0.95, tt, time)
    if wp.length(p - wp.vec3(x0, _Y, 0.0)) < _R + 0.02:
        return wp.vec3(0.28, 0.5, 0.85)                      # static: cool blue
    if wp.length(p - wp.vec3(x1, _Y, 0.0)) < _R + 0.02:
        return wp.vec3(0.85, 0.62, 0.25)                     # slow: amber
    if wp.length(p - wp.vec3(x2, _Y, 0.0)) < _R + 0.02:
        return wp.vec3(0.85, 0.28, 0.3)                      # fast: red
    if wp.length(p - _SPIN) < _R + 0.02:
        # striped sphere, spun about y by an angle that sweeps across the shutter
        q = p - _SPIN
        ang = 1.3 * (tt - 0.5) + time * 6.2831853
        ca = wp.cos(-ang)
        sa = wp.sin(-ang)
        lx = q[0] * ca - q[2] * sa
        lz = q[0] * sa + q[2] * ca
        lon = wp.atan2(lz, lx)
        band = wp.sin(lon * 6.0)
        if band > 0.0:
            return wp.vec3(0.9, 0.9, 0.92)
        return wp.vec3(0.12, 0.12, 0.14)
    return wp.vec3(0.62, 0.62, 0.64)                         # room


@wp.func
def _march(ro: wp.vec3, rd: wp.vec3, tt: float, time: float):
    t = float(0.003)
    hit = int(0)
    for _ in range(120):
        p = ro + rd * t
        d = _scene(p, tt, time)
        if d < 0.0006:
            hit = 1
            break
        t += wp.max(d * 0.85, 0.0008)
        if t > 16.0:
            break
    return t, hit


@wp.func
def _sky(rd: wp.vec3) -> wp.vec3:
    # soft studio dome: bright warm overhead, cool sides — lights the whole scene
    up = wp.clamp(rd[1] * 0.5 + 0.5, 0.0, 1.0)
    warm = wp.vec3(1.5, 1.4, 1.2)
    cool = wp.vec3(0.35, 0.4, 0.5)
    return cool + (warm - cool) * up * up


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanfov: float,
                   spp: int, seed0: int, time: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    acc = wp.vec3(0.0, 0.0, 0.0)
    pix = i * width + j

    for s in range(spp):
        rng = wp.rand_init(seed0 + pix * 9781 + s * 6151, s * 2749 + 17)
        du = wp.randf(rng) - 0.5
        dv = wp.randf(rng) - 0.5
        u = (2.0 * (float(j) + 0.5 + du) / float(width) - 1.0) * tanfov * aspect
        v = (2.0 * (float(height - 1 - i) + 0.5 + dv) / float(height) - 1.0) * tanfov
        rd = wp.normalize(fwd + right * u + up * v)
        ro = eye
        tt = wp.randf(rng)                                   # <-- the shutter instant
        throughput = wp.vec3(1.0, 1.0, 1.0)
        radiance = wp.vec3(0.0, 0.0, 0.0)

        for _b in range(_BOUNCES):
            t, hit = _march(ro, rd, tt, time)
            if hit == 0:
                radiance = radiance + wp.cw_mul(throughput, _sky(rd))
                break
            p = ro + rd * t
            n = _normal(p, tt, time)
            if wp.dot(n, rd) > 0.0:
                n = -n
            throughput = wp.cw_mul(throughput, _albedo(p, tt, time))
            ro = p + n * 0.002
            rd = onb_cosine(n, wp.randf(rng), wp.randf(rng))
        acc = acc + radiance

    img[i, j] = acc / float(spp)


def _render(width, height, time, mouse, device):
    spp = _SPP
    if width * height <= 96 * 72:
        spp = 8

    ang = float(mouse[0]) * 0.004
    eye = wp.vec3(math.sin(ang) * 0.4, 0.28, 2.9)
    tgt = wp.vec3(0.0, -0.28, 0.0)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(44.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov, spp, 44441, float(time)],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.3, preserve_hue=True)


SCENE = Scene(
    name="motion_blur",
    description="motion blur from distributed path tracing — every sample picks a random "
                "instant within the shutter, so three spheres translating at rising speeds "
                "smear from sharp to a long streak while a striped sphere's spin blurs its "
                "bands, all in a soft studio dome. Temporal sampling in the same integrator "
                "that does spatial anti-aliasing.",
    renderer=_render,
)
