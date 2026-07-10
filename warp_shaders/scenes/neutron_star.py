"""Neutron star — an NVIDIA Warp port of the GLSL Shadertoy scene.

A dense pulsar core with relativistic jets along the magnetic axis, magnetic
field rings, orbiting matter, and a cube-mapped starfield. The original shader
lives at ``reference/neutron-star.frag``; this is a faithful ``@wp.kernel``
translation, exposed through the scene registry as ``SCENE``.
"""

import warp as wp

from ..scene import Scene
from ..sdf import fbm2d, hash2d, noise2d, rot2, sd_torus

# Raymarch tuning (mirrors the GLSL #defines).
STEPS = 100
MAX_DIST = 100.0
SURF_DIST = 0.001

# Material ids returned by the SDF, matching the shader's `mat` slots.
MAT_PLANET = 1.0
MAT_FIELD = 2.0
MAT_PROBE = 3.0


@wp.func
def stolb(p: wp.vec3) -> float:
    """The thin vertical plasma jet through the poles."""
    xy = rot2(wp.vec2(p[0], p[1]), 0.2)
    radius = 0.003 + wp.abs(xy[1]) * 0.02
    return wp.length(wp.vec2(xy[0], p[2])) - radius


@wp.func
def get_dist(p: wp.vec3, time: float):
    """Scene SDF. Returns (distance, material_id)."""
    speed = 0.9

    # Planet space: rotate the whole scene around Y over time.
    rot_xz = rot2(wp.vec2(p[0], p[2]), time * speed)
    p = wp.vec3(rot_xz[0], p[1], rot_xz[1])
    rotated_space = p  # already rotated; reused for the field rings below

    n_pos = wp.normalize(p)
    uv_sphere = wp.vec2(wp.atan2(n_pos[2], n_pos[0]), n_pos[1])

    q_noise = wp.vec2(
        noise2d(wp.vec2(uv_sphere[0] * 4.0 + time * 0.5, uv_sphere[1] * 4.0 + time * 0.5)),
        noise2d(wp.vec2(uv_sphere[0] * 4.0 - time * 0.3, uv_sphere[1] * 4.0 - time * 0.3)),
    )
    n = fbm2d(
        wp.vec2(
            uv_sphere[0] * 20.0 + q_noise[0] * 1.5 + time * 0.4,
            uv_sphere[1] * 20.0 + q_noise[1] * 1.5 + time * 0.4,
        )
    ) * 0.04

    planet = wp.length(p) - (0.2 + n)
    jet = stolb(p)
    dist = wp.min(planet, jet)

    # Field space: a tilted frame the rings + probes live in.
    field = rotated_space
    yz = rot2(wp.vec2(field[1], field[2]), 0.01)
    field = wp.vec3(field[0], yz[0], yz[1])
    xy = rot2(wp.vec2(field[0], field[1]), 0.2)
    field = wp.vec3(xy[0], xy[1], field[2])

    fields = float(1.0e5)
    probes = float(1.0e5)

    rings = 8
    for i in range(rings):
        q = field
        a = float(i) * (6.14159 / float(rings))
        qxz = rot2(wp.vec2(q[0], q[2]), a)
        q = wp.vec3(qxz[0], q[1], qxz[1])
        qyz = rot2(wp.vec2(q[1], q[2]), 1.6)
        q = wp.vec3(q[0], qyz[0], qyz[1])
        qxz = rot2(wp.vec2(q[0], q[2]), -0.02)
        q = wp.vec3(qxz[0], q[1], qxz[1])

        qt = wp.vec3(q[0] * 0.6, q[1], q[2] * 0.5)

        t = time * 5.0 + float(i) * 1.0
        sphere_pos = wp.vec3(
            (-0.17 + wp.cos(t) * 0.2) / 0.6,
            0.0,
            (wp.sin(t) * 0.2) / 0.5,
        )

        probe = wp.length(q - sphere_pos) - 0.004
        magnit = sd_torus(qt + wp.vec3(0.17, 0.0, 0.0), wp.vec2(0.2, 0.0001))

        fields = wp.min(fields, magnit)
        probes = wp.min(probes, probe)

    mat = MAT_PLANET
    if fields < dist:
        dist = fields
        mat = MAT_FIELD
    if probes < dist:
        dist = probes
        mat = MAT_PROBE

    dist = wp.min(dist, fields)
    return dist, mat


@wp.func
def raymarch(ro: wp.vec3, rd: wp.vec3, time: float):
    """March the scene. Returns (distance, accumulated_glow, hit_material)."""
    d = float(0.0)
    glow = float(0.0)
    mat = MAT_PLANET
    for _ in range(STEPS):
        p = ro + rd * d
        ds, m = get_dist(p, time)
        mat = m
        d += ds
        if mat == MAT_PLANET:
            glow += 1.0 / (wp.abs(ds) + 0.022)
        if mat == MAT_FIELD:
            glow += 1.0 / (wp.abs(ds) + 0.19)
        if mat == MAT_PROBE:
            glow += 1.0 / (wp.abs(ds) + 0.019)
        if d > MAX_DIST or wp.abs(ds) < SURF_DIST:
            break
    return d, glow, mat


@wp.func
def camera_ray(uv: wp.vec2, ro: wp.vec3, target: wp.vec3, zoom: float) -> wp.vec3:
    """Build a look-at ray direction (the GLSL ``R()`` helper)."""
    f = wp.normalize(target - ro)
    r = wp.normalize(wp.cross(wp.vec3(0.0, 1.0, 0.0), f))
    u = wp.cross(f, r)
    c = ro + f * zoom
    i = c + r * uv[0] + u * uv[1]
    return wp.normalize(i - ro)


@wp.func
def cube_stars(rd: wp.vec3) -> float:
    """Procedural starfield projected onto the dominant cube face."""
    ax = wp.abs(rd[0])
    ay = wp.abs(rd[1])
    az = wp.abs(rd[2])
    max_axis = wp.max(ax, wp.max(ay, az))

    uv = wp.vec2(0.0, 0.0)
    sector = wp.vec3(0.0, 0.0, 0.0)
    if max_axis == ax:
        uv = wp.vec2(rd[1] / rd[0], rd[2] / rd[0])
        sector = wp.vec3(wp.sign(rd[0]), 0.0, 0.0)
    elif max_axis == ay:
        uv = wp.vec2(rd[0] / rd[1], rd[2] / rd[1])
        sector = wp.vec3(0.0, wp.sign(rd[1]), 0.0)
    else:
        uv = wp.vec2(rd[0] / rd[2], rd[1] / rd[2])
        sector = wp.vec3(0.0, 0.0, wp.sign(rd[2]))

    p = wp.vec2(uv[0] * 75.0, uv[1] * 75.0)
    cell = wp.vec2(wp.floor(p[0]), wp.floor(p[1]))
    gv = wp.vec2(p[0] - cell[0] - 0.5, p[1] - cell[1] - 0.5)

    final_id = wp.vec2(cell[0] + sector[0] + sector[2] * 15.1, cell[1] + sector[1] + sector[2] * 15.1)
    h = hash2d(final_id)

    star = float(0.0)
    if h > 0.88:
        offset = wp.vec2(
            (hash2d(wp.vec2(final_id[0] + 0.35, final_id[1] + 0.35)) - 0.5) * 0.6,
            (hash2d(wp.vec2(final_id[0] + 0.72, final_id[1] + 0.72)) - 0.5) * 0.6,
        )
        d = wp.length(wp.vec2(gv[0] - offset[0], gv[1] - offset[1]))
        size = 0.2 + hash2d(wp.vec2(final_id[0] * 1.5, final_id[1] * 1.5)) * 0.08
        star = wp.exp(-35.0 * d / size)
    return star


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()  # i = row (0 = top), j = column

    # GLSL fragCoord is bottom-up, pixel-centered.
    fx = float(j) + 0.5
    fy = float(height - 1 - i) + 0.5

    res = wp.vec2(float(width), float(height))
    uv = wp.vec2((fx - 0.5 * res[0]) / res[1], (fy - 0.5 * res[1]) / res[1])
    m = wp.vec2(mouse[0] / res[0], mouse[1] / res[1])

    ro = wp.vec3(0.0, -0.1, 2.0)
    yz = rot2(wp.vec2(ro[1], ro[2]), -m[1] * 3.14 + 1.0)
    ro = wp.vec3(ro[0], yz[0], yz[1])
    xz = rot2(wp.vec2(ro[0], ro[2]), -m[0] * 6.2831)
    ro = wp.vec3(xz[0], ro[1], xz[1])

    target = wp.vec3(0.0, 0.01, 0.0)
    rd = camera_ray(uv, ro, target, 1.2)

    star = cube_stars(rd)
    stars = wp.vec3(0.7, 0.85, 1.0) * (star * 100.5)
    stars = wp.vec3(stars[0] * 0.1, stars[1] * 0.5, stars[2] * 1.0)

    d, glow, mat = raymarch(ro, rd, time)

    base = wp.vec3(0.2, 0.4, 0.7) * (glow * 0.0045)
    col = wp.vec3(base[0] + 0.09, base[1] + 0.09, base[2] + 0.09)

    if d < 50.0:
        col = wp.vec3(0.2, 0.5, 0.7) * (glow * 0.02)
    else:
        col = col + stars

    img[i, j] = col


SCENE = Scene(
    name="neutron_star",
    kernel=render_kernel,
    description="Pulsar core with relativistic jets, magnetic field rings, orbiting matter, and a starfield.",
)
