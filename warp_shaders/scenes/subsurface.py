"""Subsurface scattering — light that random-walks *inside* a translucent solid.

The engine leap continued. A diffuse surface stops light dead; a dielectric passes it
straight through. **Subsurface scattering** is the middle case that makes wax, jade, marble
and skin look alive: light refracts into the body, **bounces around inside** the medium
(scattering off microscopic structure), loses a little energy to absorption at every step,
and eventually escapes somewhere else on the surface. Where the solid is *thin* the walk
escapes quickly and the material **glows** with the backlight; where it is *thick* the walk
is absorbed and the core goes dark and saturated.

This scene path-traces that literally: at a subsurface hit the ray enters the medium and
does a bounded **random walk** (exponential free-flight steps, isotropic scatter, a warm
single-scattering albedo multiplied in at every step) until it exits back into the room and
continues toward the light. A jade sphere and a thin standing ring sit in front of a warm
back-panel — the ring, thin everywhere, glows through; the sphere shows a bright translucent
rim around a denser, warmer core. See ``docs/research/39-engine-leap.md``.
"""

import math

import numpy as np
import warp as wp

from ..engine import post
from ..procedural.sdf import op_union, sd_box, sd_sphere, sd_torus
from ..scene import Scene

_SPP = 320
_BOUNCES = 3
_WALK = 40                       # max scatter events inside the medium
_MFP = wp.constant(0.075)        # mean free path — small vs the sphere, big vs the ring tube
_SSS_ALB = wp.constant(wp.vec3(0.94, 0.72, 0.5))   # warm single-scattering albedo (honey/jade)
_SSS_ENTER = wp.constant(wp.vec3(0.98, 0.9, 0.82))  # surface tint on entry

_SPH = wp.constant(wp.vec3(-0.5, -0.28, 0.05))
_SPHR = 0.47
_TOR = wp.constant(wp.vec3(0.52, -0.3, 0.0))


@wp.func
def _obj(p: wp.vec3) -> float:
    # the translucent body only (no room): a sphere + a thin standing ring
    s = sd_sphere(p - _SPH, _SPHR)
    q = p - _TOR
    ring = sd_torus(wp.vec3(q[0], q[2], q[1]), wp.vec2(0.4, 0.12))  # axis along z (faces camera)
    return op_union(s, ring)


@wp.func
def _scene(p: wp.vec3) -> float:
    floor = sd_box(p - wp.vec3(0.0, -0.78, 0.0), wp.vec3(2.2, 0.04, 2.2))
    back = sd_box(p - wp.vec3(0.0, 0.2, -1.45), wp.vec3(2.2, 1.3, 0.04))
    return op_union(op_union(floor, back), _obj(p))


@wp.func
def _normal(p: wp.vec3) -> wp.vec3:
    e = 0.0015
    dx = _scene(p + wp.vec3(e, 0.0, 0.0)) - _scene(p - wp.vec3(e, 0.0, 0.0))
    dy = _scene(p + wp.vec3(0.0, e, 0.0)) - _scene(p - wp.vec3(0.0, e, 0.0))
    dz = _scene(p + wp.vec3(0.0, 0.0, e)) - _scene(p - wp.vec3(0.0, 0.0, e))
    return wp.normalize(wp.vec3(dx, dy, dz))


@wp.func
def _is_sss(p: wp.vec3) -> int:
    if _obj(p) < 0.01:
        return 1
    return 0


@wp.func
def _albedo(p: wp.vec3) -> wp.vec3:
    if p[2] < -1.35:
        return wp.vec3(0.7, 0.7, 0.72)       # back panel body (its emissive strip lives on it)
    return wp.vec3(0.5, 0.5, 0.52)           # floor


@wp.func
def _emission(p: wp.vec3) -> wp.vec3:
    # warm back-panel light strip — the source that shines through the ring/sphere
    if p[2] < -1.4 and wp.abs(p[0]) < 1.15 and p[1] > -0.55 and p[1] < 0.95:
        return wp.vec3(10.0, 6.5, 3.8)
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def _march(ro: wp.vec3, rd: wp.vec3):
    t = float(0.003)
    hit = int(0)
    for _ in range(120):
        p = ro + rd * t
        d = _scene(p)
        if d < 0.0006:
            hit = 1
            break
        t += wp.max(d * 0.85, 0.0008)
        if t > 16.0:
            break
    return t, hit


@wp.func
def _onb(n: wp.vec3, r1: float, r2: float) -> wp.vec3:
    a = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(n[0]) > 0.9:
        a = wp.vec3(0.0, 1.0, 0.0)
    tang = wp.normalize(wp.cross(a, n))
    bit = wp.cross(n, tang)
    r = wp.sqrt(r1)
    phi = 6.2831853 * r2
    return wp.normalize(tang * (r * wp.cos(phi)) + bit * (r * wp.sin(phi))
                        + n * wp.sqrt(wp.max(0.0, 1.0 - r1)))


@wp.func
def _sphere_dir(r1: float, r2: float) -> wp.vec3:
    z = 1.0 - 2.0 * r1
    r = wp.sqrt(wp.max(0.0, 1.0 - z * z))
    phi = 6.2831853 * r2
    return wp.vec3(r * wp.cos(phi), r * wp.sin(phi), z)


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
        du = wp.randf(rng) - 0.5
        dv = wp.randf(rng) - 0.5
        u = (2.0 * (float(j) + 0.5 + du) / float(width) - 1.0) * tanfov * aspect
        v = (2.0 * (float(height - 1 - i) + 0.5 + dv) / float(height) - 1.0) * tanfov
        rd = wp.normalize(fwd + right * u + up * v)
        ro = eye
        throughput = wp.vec3(1.0, 1.0, 1.0)
        radiance = wp.vec3(0.0, 0.0, 0.0)

        for _b in range(_BOUNCES):
            t, hit = _march(ro, rd)
            if hit == 0:
                radiance = radiance + wp.cw_mul(throughput, wp.vec3(0.03, 0.04, 0.06))
                break
            p = ro + rd * t
            ng = _normal(p)
            emit = _emission(p)
            if emit[0] > 0.0:
                radiance = radiance + wp.cw_mul(throughput, emit)
                break
            if _is_sss(p) == 1:                              # subsurface random walk
                n = ng
                if wp.dot(n, rd) > 0.0:
                    n = -n
                throughput = wp.cw_mul(throughput, _SSS_ENTER)
                pos = p - n * 0.004                          # just inside the surface
                wdir = rd                                    # carry momentum inward
                escaped = int(0)
                for _w in range(_WALK):
                    step = -wp.log(wp.max(wp.randf(rng), 0.0001)) * _MFP
                    npos = pos + wdir * step
                    if _obj(npos) > 0.0:                     # walked out through the surface
                        pos = npos
                        escaped = 1
                        break
                    pos = npos
                    throughput = wp.cw_mul(throughput, _SSS_ALB)
                    wdir = _sphere_dir(wp.randf(rng), wp.randf(rng))   # isotropic scatter
                if escaped == 0:
                    break                                    # absorbed deep inside
                ro = pos + wdir * 0.01
                rd = wdir
            else:                                            # diffuse room
                n = ng
                if wp.dot(n, rd) > 0.0:
                    n = -n
                throughput = wp.cw_mul(throughput, _albedo(p))
                ro = p + n * 0.002
                rd = _onb(n, wp.randf(rng), wp.randf(rng))
        acc = acc + radiance

    img[i, j] = acc / float(spp)


def _render(width, height, time, mouse, device):
    spp = _SPP
    if width * height <= 96 * 72:
        spp = 6

    ang = 0.35 + float(mouse[0]) * 0.004
    eye = wp.vec3(2.35 * math.sin(ang), 0.12, 2.35 * math.cos(ang))
    tgt = wp.vec3(0.0, -0.28, -0.1)
    fwd = wp.normalize(tgt - eye)
    right = wp.normalize(wp.cross(fwd, wp.vec3(0.0, 1.0, 0.0)))
    up = wp.cross(right, fwd)
    tanfov = math.tan(math.radians(40.0) * 0.5)

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov, spp, 33331],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.6, preserve_hue=True)


SCENE = Scene(
    name="subsurface",
    description="subsurface scattering, path-traced — a jade sphere and a thin standing ring "
                "lit by a warm back-panel, with light entering the translucent medium and "
                "doing a bounded random walk (exponential free flights, isotropic scatter, a "
                "warm single-scattering albedo) before it escapes. The thin ring glows through; "
                "the sphere shows a translucent rim around a denser, warmer core.",
    renderer=_render,
)
