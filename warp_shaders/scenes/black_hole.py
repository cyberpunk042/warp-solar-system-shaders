"""Black hole — an NVIDIA Warp port of a self-contained GLSL lensing shader.

A Schwarzschild-ish black hole: rays are bent toward the singularity each step
(gravitational lensing), a hot accretion disk is raymarched with Doppler/red
shift tinting, and a procedural star + nebula background fills the rest.

The only channel input in the original is ``iChannel0`` (a nebula texture),
substituted here with procedural value-noise fBm per the gallery convention, so
the scene needs no assets. ``iMouse`` drives zoom (x) and pitch (y).

Original GLSL kept at ``reference/black-hole.frag``.
"""

import warp as wp

from ..sdf import fract
from ..scene import Scene

# Scene constants (the GLSL #defines).
SIZE = 0.3      # _Size: size of the black hole
SPEED = 3.0     # _Speed: disk rotation speed
STEPS = 12      # _Steps: disk texture layers


@wp.func
def mix3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:
    return wp.vec3(a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t)


@wp.func
def cmul3(a: wp.vec3, b: wp.vec3) -> wp.vec3:
    return wp.vec3(a[0] * b[0], a[1] * b[1], a[2] * b[2])


@wp.func
def sq3(v: wp.vec3) -> wp.vec3:
    return wp.vec3(v[0] * v[0], v[1] * v[1], v[2] * v[2])


@wp.func
def hash1(x: float) -> float:
    return fract(wp.sin(x) * 152754.742)


@wp.func
def hash2(p: wp.vec2) -> float:
    return hash1(p[0] + hash1(p[1]))


@wp.func
def value(p: wp.vec2, f: float) -> float:
    """Value noise at frequency ``f`` (GLSL ``value(p, f)``)."""
    qx = p[0] * f
    qy = p[1] * f
    bl = hash2(wp.vec2(wp.floor(qx + 0.0), wp.floor(qy + 0.0)))
    br = hash2(wp.vec2(wp.floor(qx + 1.0), wp.floor(qy + 0.0)))
    tl = hash2(wp.vec2(wp.floor(qx + 0.0), wp.floor(qy + 1.0)))
    tr = hash2(wp.vec2(wp.floor(qx + 1.0), wp.floor(qy + 1.0)))
    frx = fract(qx)
    fry = fract(qy)
    frx = (3.0 - 2.0 * frx) * frx * frx
    fry = (3.0 - 2.0 * fry) * fry * fry
    b = wp.lerp(bl, br, frx)
    t = wp.lerp(tl, tr, frx)
    return wp.lerp(b, t, fry)


@wp.func
def vfbm(u: float, v: float) -> float:
    n = 0.5 * value(wp.vec2(u, v), 3.0)
    n += 0.25 * value(wp.vec2(u, v), 6.0)
    n += 0.125 * value(wp.vec2(u, v), 12.0)
    n += 0.0625 * value(wp.vec2(u, v), 24.0)
    return n


@wp.func
def nebula_tex(u: float, v: float) -> wp.vec3:
    """Procedural stand-in for iChannel0 (the nebula texture)."""
    return wp.vec3(vfbm(u, v), vfbm(u + 3.3, v + 1.7), vfbm(u + 7.1, v + 9.2))


@wp.func
def background(ray: wp.vec3) -> wp.vec3:
    ux = ray[0]
    uy = ray[1]
    if wp.abs(ray[0]) > 0.5:
        ux = ray[2]
    elif wp.abs(ray[1]) > 0.5:
        uy = ray[2]

    brightness = value(wp.vec2(ux * 3.0, uy * 3.0), 100.0)
    color = value(wp.vec2(ux * 2.0, uy * 2.0), 20.0)
    brightness = wp.pow(brightness, 256.0) * 100.0
    brightness = wp.clamp(brightness, 0.0, 1.0)
    stars = mix3(wp.vec3(1.0, 0.6, 0.2), wp.vec3(0.2, 0.6, 1.0), color) * brightness

    neb = nebula_tex(ux * 1.5, uy * 1.5)
    s = neb[0] + neb[1] + neb[2]
    n = wp.vec3((neb[0] + s) * 0.25, (neb[1] + s) * 0.25, (neb[2] + s) * 0.25)
    n = sq3(n)
    n = sq3(n)
    n = sq3(n)
    n = sq3(n)  # ^16 contrast, as in the original
    return n + stars


@wp.func
def rotate3(v: wp.vec3, ax: float, ay: float) -> wp.vec3:
    """GLSL ``Rotate(v, vec2(ax, ay))`` — yz then xz, applied in that order."""
    vy = v[1]
    vz = v[2]
    ny = wp.cos(ay) * vy - wp.sin(ay) * vz
    nz = wp.cos(ay) * vz + wp.sin(ay) * vy
    vx = v[0]
    nx = wp.cos(ax) * vx - wp.sin(ax) * nz
    nz2 = wp.cos(ax) * nz + wp.sin(ax) * vx
    return wp.vec3(nx, ny, nz2)


@wp.func
def raymarch_disk(ray: wp.vec3, zero_pos: wp.vec3, time: float):
    """Raymarch the accretion disk. Returns (rgb, alpha)."""
    size = 0.3
    steps = 12.0

    position = zero_pos
    length_pos = wp.length(wp.vec2(position[0], position[2]))
    dist = wp.min(1.0, length_pos * (1.0 / size) * 0.5) * size * 0.4 * (1.0 / steps) / wp.abs(ray[1])
    position = position + ray * (dist * steps * 0.5)

    dpx = -zero_pos[2] * 0.01 + zero_pos[0]
    dpy = zero_pos[0] * 0.01 + zero_pos[2]
    delta = wp.normalize(wp.vec2(dpx - zero_pos[0], dpy - zero_pos[2]))

    parallel = ray[0] * delta[0] + ray[2] * delta[1]
    parallel /= wp.sqrt(length_pos)
    parallel *= 0.5
    red_shift = parallel + 0.3
    red_shift *= red_shift
    red_shift = wp.clamp(red_shift, 0.0, 1.0)

    dis_mix = wp.clamp((length_pos - size * 2.0) * (1.0 / size) * 0.24, 0.0, 1.0)
    inside = mix3(wp.vec3(1.0, 0.8, 0.0), wp.vec3(0.1, 0.026, 0.004), dis_mix)
    inside = cmul3(inside, mix3(wp.vec3(0.4, 0.2, 0.1), wp.vec3(1.6, 2.4, 4.0), red_shift))
    inside = inside * 1.25
    red_shift += 0.12
    red_shift *= red_shift

    o_rgb = wp.vec3(0.0, 0.0, 0.0)
    o_a = float(0.0)

    for k in range(12):
        fi = float(k)
        position = position - ray * dist
        intensity = wp.clamp(1.0 - wp.abs((fi - 0.8) * (1.0 / steps) * 2.0), 0.0, 1.0)
        length_pos = wp.length(wp.vec2(position[0], position[2]))
        dist_mult = wp.clamp((length_pos - size * 0.75) * (1.0 / size) * 1.5, 0.0, 1.0)
        dist_mult *= wp.clamp((size * 10.0 - length_pos) * (1.0 / size) * 0.20, 0.0, 1.0)
        dist_mult *= dist_mult

        u = length_pos + time * size * 0.3 + intensity * size * 0.2
        rot = time * SPEED - wp.floor(time * SPEED / 8192.0) * 8192.0  # mod(time*speed, 8192)
        xyx = -position[2] * wp.sin(rot) + position[0] * wp.cos(rot)
        xyy = position[0] * wp.sin(rot) + position[2] * wp.cos(rot)
        x = wp.abs(xyx / (xyy + 1.0e-8))
        angle = 0.02 * wp.atan(x)

        f = 70.0
        noise = value(wp.vec2(angle, u * (1.0 / size) * 0.05), f)
        noise = noise * 0.66 + 0.33 * value(wp.vec2(angle, u * (1.0 / size) * 0.05), f * 2.0)

        extra_width = noise * (1.0 - wp.clamp(fi * (1.0 / steps) * 2.0 - 1.0, 0.0, 1.0))
        alpha = wp.clamp(noise * (intensity + extra_width) * ((1.0 / size) * 10.0 + 0.01) * dist * dist_mult, 0.0, 1.0)

        col = mix3(cmul3(wp.vec3(0.3, 0.2, 0.15), inside), inside, wp.min(1.0, intensity * 2.0)) * 2.0
        o_rgb = wp.vec3(
            wp.clamp(col[0] * alpha + o_rgb[0] * (1.0 - alpha), 0.0, 1.0),
            wp.clamp(col[1] * alpha + o_rgb[1] * (1.0 - alpha), 0.0, 1.0),
            wp.clamp(col[2] * alpha + o_rgb[2] * (1.0 - alpha), 0.0, 1.0),
        )
        o_a = wp.clamp(o_a * (1.0 - alpha) + alpha, 0.0, 1.0)

        length_pos *= (1.0 / size)
        add = red_shift * (intensity + 0.5) * (1.0 / steps) * 100.0 * dist_mult / (length_pos * length_pos)
        o_rgb = o_rgb + wp.vec3(add, add, add)

    o_rgb = wp.vec3(
        wp.clamp(o_rgb[0] - 0.005, 0.0, 1.0),
        wp.clamp(o_rgb[1] - 0.005, 0.0, 1.0),
        wp.clamp(o_rgb[2] - 0.005, 0.0, 1.0),
    )
    return o_rgb, o_a


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()
    size = 0.3
    res = wp.vec2(float(width), float(height))

    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5

    frx = fx * 0.985 + fy * 0.174
    fry = fy * 0.985 - fx * 0.174
    frx += -0.06 * res[0]
    fry += 0.12 * res[1]

    rx = (frx - res[0] * 0.5) / res[0]
    ry = (fry - res[1] * 0.5) / res[0]
    ray = wp.normalize(wp.vec3(rx, ry, 1.0))

    # Camera: mouse.x -> zoom distance, mouse.y -> pitch. Default (0,0) -> pos.z = -5.
    mz = 20.0 * mouse[0] / res[1] - 10.0
    pos = wp.vec3(0.0, 0.05, -(mz * mz) * 0.05)

    ax0 = time * 0.1
    ay0 = (2.0 * mouse[1] / res[1]) * 3.14 + 0.1 + 3.14
    dist = wp.length(pos)
    pos = rotate3(pos, ax0, ay0)
    sub = wp.min(0.3 / dist, 3.14)
    ray = rotate3(ray, ax0 - sub, ay0 - sub * 0.5)

    col_rgb = wp.vec3(0.0, 0.0, 0.0)
    col_a = float(0.0)
    glow_rgb = wp.vec3(0.0, 0.0, 0.0)
    glow_a = float(0.0)
    out_rgb = wp.vec3(0.0, 0.0, 0.0)
    escaped = int(0)

    for _ in range(20):
        for _h in range(6):
            dotpos = wp.dot(pos, pos)
            inv_dist = 1.0 / wp.sqrt(dotpos)
            cent_dist = dotpos * inv_dist
            step_dist = 0.92 * wp.abs(pos[1] / ray[1])
            far_limit = cent_dist * 0.5
            close_limit = cent_dist * 0.1 + 0.05 * cent_dist * cent_dist * (1.0 / size)
            step_dist = wp.min(step_dist, wp.min(far_limit, close_limit))

            inv_dist_sqr = inv_dist * inv_dist
            bend = step_dist * inv_dist_sqr * size * 0.625
            ray = wp.normalize(ray - pos * (bend * inv_dist))
            pos = pos + ray * step_dist

            g = 0.01 * step_dist * inv_dist_sqr * inv_dist_sqr * wp.clamp(cent_dist * 2.0 - 1.2, 0.0, 1.0)
            glow_rgb = glow_rgb + wp.vec3(1.2, 1.1, 1.0) * g
            glow_a = glow_a + g

        dist2 = wp.length(pos)
        if dist2 < size * 0.1:  # swallowed
            out_rgb = col_rgb * col_a + glow_rgb * (1.0 - col_a)
            escaped = 1
            break
        if dist2 > size * 1000.0:  # escaped to background
            bg = background(ray)
            out_rgb = col_rgb * col_a + bg * (1.0 - col_a) + glow_rgb * (1.0 - col_a)
            escaped = 1
            break
        if wp.abs(pos[1]) <= size * 0.002:  # crossed the disk plane
            drgb, da = raymarch_disk(ray, pos, time)
            pos[1] = 0.0
            pos = pos + ray * wp.abs(size * 0.001 / ray[1])
            col_rgb = drgb * (1.0 - col_a) + col_rgb
            col_a = col_a + da * (1.0 - col_a)

    if escaped == 0:
        out_rgb = col_rgb + glow_rgb * (col_a + glow_a)

    out_rgb = wp.vec3(
        wp.pow(wp.max(out_rgb[0], 0.0), 0.6),
        wp.pow(wp.max(out_rgb[1], 0.0), 0.6),
        wp.pow(wp.max(out_rgb[2], 0.0), 0.6),
    )
    img[i, j] = out_rgb


SCENE = Scene(
    name="black_hole",
    kernel=render_kernel,
    description="Gravitationally-lensed black hole with a raymarched accretion disk. iMouse: zoom (x) / pitch (y).",
)
