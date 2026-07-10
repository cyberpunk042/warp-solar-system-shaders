"""Planet — an NVIDIA Warp port of a lit-planet / star / lens-flare GLSL scene
(an iq + Antonalog + mu6k + reinder mashup from Shadertoy).

A ray-vs-sphere planet with a domain-warped ("distort") procedural surface, lit
by a distant star that also casts a lens flare and glints; a cube-mapped
procedural starfield fills the background.

The only channel input in the original is ``iChannel0`` (a texture sampled by
column for surface color), replaced here with an IQ cosine palette per the
gallery convention. ``iMouse`` orbits the camera (drag).

Original GLSL kept at ``reference/planet.frag``.
"""

import warp as wp

from ..sdf import fract
from ..scene import Scene


@wp.func
def smoothstep_(e0: float, e1: float, x: float) -> float:
    t = wp.clamp((x - e0) / (e1 - e0 + 1.0e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@wp.func
def hash(n: float) -> float:
    return fract(wp.sin(n) * 123.456789)


@wp.func
def rotate(uv: wp.vec2, a: float) -> wp.vec2:
    c = wp.cos(a)
    s = wp.sin(a)
    return wp.vec2(c * uv[0] - s * uv[1], s * uv[0] + c * uv[1])


@wp.func
def noise3(p: wp.vec3) -> float:
    """iq value noise over a 3D lattice."""
    flx = wp.floor(p[0])
    fly = wp.floor(p[1])
    flz = wp.floor(p[2])
    frx = fract(p[0])
    fry = fract(p[1])
    frz = fract(p[2])
    frx = frx * frx * (3.0 - 2.0 * frx)
    fry = fry * fry * (3.0 - 2.0 * fry)
    frz = frz * frz * (3.0 - 2.0 * frz)

    n = flx + fly * 157.0 + 113.0 * flz
    return wp.lerp(
        wp.lerp(
            wp.lerp(hash(n + 0.0), hash(n + 1.0), frx),
            wp.lerp(hash(n + 157.0), hash(n + 158.0), frx), fry),
        wp.lerp(
            wp.lerp(hash(n + 113.0), hash(n + 114.0), frx),
            wp.lerp(hash(n + 270.0), hash(n + 271.0), frx), fry),
        frz)


@wp.func
def fbm(p: wp.vec2, t: float) -> float:
    f = 0.5 * noise3(wp.vec3(p[0], p[1], t))
    p = p * 2.1
    f += 0.25 * noise3(wp.vec3(p[0], p[1], t))
    p = p * 2.2
    f += 0.125 * noise3(wp.vec3(p[0], p[1], t))
    p = p * 2.3
    f += 0.0625 * noise3(wp.vec3(p[0], p[1], t))
    return f


@wp.func
def field(p: wp.vec2, time: float) -> wp.vec2:
    t = 0.2 * time  # time_scale
    p = wp.vec2(p[0] + t, p[1])
    n = fbm(p, t)
    e = 0.25
    nx = fbm(wp.vec2(p[0] + e, p[1]), t)
    ny = fbm(wp.vec2(p[0], p[1] + e), t)
    return wp.vec2(n - ny, nx - n) / e


@wp.func
def palette(t: float) -> wp.vec3:
    """IQ cosine palette standing in for the iChannel0 surface texture."""
    d = wp.vec3(0.0, 0.33, 0.67)
    return wp.vec3(
        0.5 + 0.5 * wp.cos(6.28318 * (t + d[0])),
        0.5 + 0.5 * wp.cos(6.28318 * (t + d[1])),
        0.5 + 0.5 * wp.cos(6.28318 * (t + d[2])),
    )


@wp.func
def distort(p: wp.vec2, time: float) -> wp.vec3:
    for _ in range(5):  # distort_iterations
        p = p + field(p, time) / 5.0
    s = palette(p[1] * 0.025) * 2.5  # tex_scale; procedural surface color
    return s * fbm(p, 0.0)


@wp.func
def surface_uv(uvx: float, uvy: float, time: float) -> wp.vec2:
    return wp.vec2(uvx * 5.0 + 0.01 * time, uvy * 15.0)  # map()


@wp.func
def background_stars(dir: wp.vec3, time: float) -> wp.vec3:
    ax = wp.abs(dir[0])
    ay = wp.abs(dir[1])
    az = wp.abs(dir[2])
    uv = wp.vec2(0.0, 0.0)
    if ax > ay and ax > az:
        uv = wp.vec2(dir[1] / dir[0], dir[2] / dir[0])
    elif ay > ax and ay > az:
        uv = wp.vec2(dir[2] / dir[1], dir[0] / dir[1])
    else:
        uv = wp.vec2(dir[0] / dir[2], dir[1] / dir[2])

    f = float(0.0)
    for _ in range(10):  # star_iterations
        uv = rotate(wp.vec2(1.07 * uv[0] + 0.7, 1.07 * uv[1] + 0.7), 0.5)
        t = 10.0 * uv[0] * uv[1] + time
        fv = fbm(wp.vec2(10.0 * uv[0], 10.0 * uv[1]), 0.0)
        ux = wp.cos(100.0 * uv[0]) * fv
        uy = wp.cos(100.0 * uv[1]) * fv
        f += smoothstep_(0.5, 0.55, ux * uy) * (0.25 * wp.sin(t) + 0.75)
    return wp.vec3(1.0, 0.7, 0.5) * f


@wp.func
def main_star(uv: wp.vec2, sp: wp.vec2, time: float) -> wp.vec3:
    t = wp.atan2(uv[0] - sp[0], uv[1] - sp[1])
    n = 2.0 + noise3(wp.vec3(10.0 * t, time, 0.0))
    d = wp.max(wp.length(uv - sp) * 25.0, 1.0e-3)
    return wp.vec3(1.0, 0.7, 0.5) * ((1.0 + n) / (d * d * d))


@wp.func
def cast_sphere(p: wp.vec3, rd: wp.vec3) -> float:
    b = wp.dot(p, rd)
    c = wp.dot(p, p) - 1.0
    f = b * b - c
    if f >= 0.0:
        return -b - wp.sqrt(f)
    return -1.0


@wp.func
def material(pos: wp.vec3, time: float) -> wp.vec3:
    uvx = wp.atan2(pos[0], pos[2])
    uvy = wp.asin(wp.clamp(pos[1], -1.0, 1.0))
    return distort(surface_uv(uvx, uvy, time), time)


@wp.func
def lighting(n: wp.vec3, c: wp.vec3, rd: wp.vec3, rdc: wp.vec3) -> wp.vec3:
    pos_star = wp.vec3(0.0, 9.0, 30.0)
    col_star = wp.vec3(1.0, 0.7, 0.5)
    l = wp.normalize(pos_star + (pos_star - rdc * wp.dot(pos_star, rdc)) * 2.0)
    ndl = wp.dot(n, l)
    ndr = wp.dot(n, -rd)
    ldr = wp.dot(l, rd)
    f = wp.max(ndl, 0.0) + 0.002
    g = ldr * smoothstep_(0.0, 0.1, ndr) * wp.pow(wp.max(1.0 - ndr, 0.0), 10.0)
    r = c * f + col_star * g
    return wp.vec3(wp.clamp(r[0], 0.0, 1.0), wp.clamp(r[1], 0.0, 1.0), wp.clamp(r[2], 0.0, 1.0))


@wp.func
def flare(uv: wp.vec2, dir: wp.vec2, s: float) -> float:
    proj = wp.dot(uv, dir)
    d = wp.length(uv - dir * proj)
    lu = wp.length(uv)
    base = wp.max(1.0 - d, 0.0)
    f = wp.max(wp.pow(base, 128.0) * (1.0 * s - lu), 0.0)
    f += wp.max(wp.pow(base, 64.0) * (0.5 * s - lu), 0.0)
    f += wp.max(wp.pow(base, 32.0) * (0.25 * s - lu), 0.0)
    f += wp.max(wp.pow(base, 16.0) * (0.125 * s - lu), 0.0)
    return f


@wp.func
def lens_glint(uv: wp.vec2, c: wp.vec2, r: float, w: float) -> float:
    l = wp.length(uv - c)
    return wp.length(c) * smoothstep_(0.0, w * r, l) * (1.0 - smoothstep_(w * r, r, l))


@wp.func
def render_scene(uv: wp.vec2, campos: wp.vec3, uu: wp.vec3, vv: wp.vec3, ww: wp.vec3, time: float) -> wp.vec3:
    pos_star = wp.vec3(0.0, 9.0, 30.0)
    col_star = wp.vec3(1.0, 0.7, 0.5)

    vdir = wp.normalize(wp.vec3(uv[0], uv[1], 1.0))
    rd = uu * vdir[0] + vv * vdir[1] + ww * vdir[2]  # m * vdir
    rdc = ww

    c = background_stars(rd, time)

    rel = pos_star - campos
    cp = wp.vec3(wp.dot(uu, rel), wp.dot(vv, rel), wp.dot(ww, rel))  # mInv * rel
    cpz = cp[2]
    spx = float(0.0)
    spy = float(0.0)
    if cpz > 0.0:
        spx = cp[0] / cpz
        spy = cp[1] / cpz
        c = c + main_star(uv, wp.vec2(spx, spy), time)

    t = cast_sphere(campos, rd)
    if t > 0.0:
        pos = campos + rd * t
        nor = wp.normalize(pos)
        c = material(pos, time)
        c = lighting(nor, c, rd, rdc)

    if cpz > 0.0 and spx > -1.0 and spx < 1.0:
        sp = wp.vec2(spx, spy)
        oc = smoothstep_(0.35, 0.4, wp.length(sp))
        f = flare(uv - sp, wp.vec2(1.0, 0.0), oc)
        f += oc * 0.05 * lens_glint(uv, sp * -0.4, 0.2, 0.92)
        f += oc * 0.09 * lens_glint(uv, sp * -0.8, 0.3, 0.95)
        f += oc * 0.04 * lens_glint(uv, sp * -1.1, 0.06, 0.8)
        c = c + col_star * f

    return c


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()
    res = wp.vec2(float(width), float(height))

    uvx = (float(j) + 0.5) / res[0] - 0.5
    uvy = (float(height - 1 - i) + 0.5) / res[1] - 0.5
    uvx *= res[0] / res[1]
    uv = wp.vec2(uvx, uvy)

    # Camera: default slow orbit; mouse.x/y take over when dragging.
    mx = mouse[0] / res[0]
    my = mouse[1] / res[1]
    cx = wp.cos(0.1 * time + 3.55)
    sx = wp.sin(0.1 * time + 3.55)
    cy = float(0.0)
    if mouse[0] > 0.0 or mouse[1] > 0.0:
        cx = wp.cos(10.0 * mx)
        sx = wp.sin(10.0 * mx)
        cy = wp.cos(3.2 * my)

    campos = wp.vec3((cx - sx) * 2.0, cy * 2.0, (sx + cx) * 2.0)
    camdir = wp.normalize(-campos)

    # Camera basis (doCamera): ww = dir, uu = norm(cross(ww, up)), vv = norm(cross(uu, ww)).
    world_up = wp.vec3(0.0, 1.0, 0.0)
    ww = camdir
    uu = wp.normalize(wp.cross(ww, world_up))
    vv = wp.normalize(wp.cross(uu, ww))

    c = render_scene(uv, campos, uu, vv, ww, time)
    c = wp.vec3(
        wp.pow(wp.max(c[0], 0.0), 0.4545),
        wp.pow(wp.max(c[1], 0.0), 0.4545),
        wp.pow(wp.max(c[2], 0.0), 0.4545),
    )
    img[i, j] = c


SCENE = Scene(
    name="planet",
    kernel=render_kernel,
    description="Lit planet with a distant star, lens flare, and cube-mapped background stars (iq/mu6k). iMouse orbits.",
)
