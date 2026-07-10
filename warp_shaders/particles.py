"""Particle primitives — the bottom of the 'atom from the bottom up' build.

Reusable Warp ``@wp.func`` building blocks shared by the particle scenes:
color-charged quarks, gluon flux tubes that bind them, a nucleon (3 bound
quarks), plus camera + volumetric helpers. Higher scenes (proton, neutron,
atom) compose these; each is stylized but structurally faithful — correct quark
content, QCD color charge summing to color-neutral, gluon confinement.

Not to physical scale (a real nucleus is ~1e-5 of the atom); sizes are chosen so
the structure is visible.
"""

import warp as wp

from .sdf import fract


# ---------------------------------------------------------------- noise
@wp.func
def hash1(n: float) -> float:
    return fract(wp.sin(n) * 43758.5453)


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
        wp.lerp(wp.lerp(hash1(n + 0.0), hash1(n + 1.0), frx),
                wp.lerp(hash1(n + 157.0), hash1(n + 158.0), frx), fry),
        wp.lerp(wp.lerp(hash1(n + 113.0), hash1(n + 114.0), frx),
                wp.lerp(hash1(n + 270.0), hash1(n + 271.0), frx), fry),
        frz)


# ---------------------------------------------------------------- camera
@wp.func
def orbit_ro(time: float, mouse: wp.vec2, res: wp.vec2, dist: float) -> wp.vec3:
    """Camera position: slow auto-orbit; mouse takes over azimuth/elevation."""
    az = 0.25 * time + 0.6
    el = 0.45
    if mouse[0] > 0.0 or mouse[1] > 0.0:
        az = (mouse[0] / res[0]) * 6.2831
        el = (mouse[1] / res[1] - 0.5) * 3.0
    ce = wp.cos(el)
    return wp.vec3(dist * ce * wp.sin(az), dist * wp.sin(el), dist * ce * wp.cos(az))


@wp.func
def camera_ray(uv: wp.vec2, ro: wp.vec3, target: wp.vec3, zoom: float) -> wp.vec3:
    ww = wp.normalize(target - ro)
    uu = wp.normalize(wp.cross(ww, wp.vec3(0.0, 1.0, 0.0)))
    vv = wp.normalize(wp.cross(uu, ww))
    vdir = wp.normalize(wp.vec3(uv[0], uv[1], zoom))
    return uu * vdir[0] + vv * vdir[1] + ww * vdir[2]


@wp.func
def ray_sphere(ro: wp.vec3, rd: wp.vec3, center: wp.vec3, radius: float):
    """Returns (t_near, t_far, hit) for a ray vs sphere."""
    oc = ro - center
    b = wp.dot(oc, rd)
    c = wp.dot(oc, oc) - radius * radius
    h = b * b - c
    if h < 0.0:
        return 0.0, 0.0, 0
    h = wp.sqrt(h)
    return -b - h, -b + h, 1


# ---------------------------------------------------------------- emitters
@wp.func
def emitter(ro: wp.vec3, rd: wp.vec3, center: wp.vec3, size: float) -> float:
    """Glow of a point emitter as seen along a ray: bright core + soft halo."""
    oc = center - ro
    t = wp.max(wp.dot(oc, rd), 0.0)
    d = wp.length(center - (ro + rd * t))
    x = d / size
    return wp.exp(-x * x * 9.0) + 0.22 * wp.exp(-x * x * 1.8)


@wp.func
def color_charge(k: int) -> wp.vec3:
    """QCD color charge: red / green / blue. The three sum to color-neutral."""
    if k == 0:
        return wp.vec3(1.0, 0.25, 0.25)
    if k == 1:
        return wp.vec3(0.3, 1.0, 0.35)
    return wp.vec3(0.35, 0.5, 1.0)


# ---------------------------------------------------------------- quarks + gluons
@wp.func
def quark_pos(k: int, time: float, conf: float) -> wp.vec3:
    """Position of quark ``k`` confined in a nucleon: rotating triad + jitter."""
    ang = float(k) * 2.0944 + time * 0.6
    jit = 0.15 * noise3(wp.vec3(time * 0.8, float(k) * 3.1, 0.0))
    rad = conf * (0.5 + jit)
    return wp.vec3(rad * wp.cos(ang), rad * wp.sin(ang), conf * 0.3 * wp.sin(time * 1.1 + float(k) * 2.1))


@wp.func
def flux(ro: wp.vec3, rd: wp.vec3, a: wp.vec3, b: wp.vec3, time: float) -> float:
    """Gluon flux tube binding two quarks: sampled line of emitters, flowing."""
    f = float(0.0)
    for s in range(8):
        u = (float(s) + 0.5) / 8.0
        p = a + (b - a) * u
        f += emitter(ro, rd, p, 0.09) * (0.6 + 0.4 * wp.sin(20.0 * u - time * 6.0))
    return f / 8.0


@wp.func
def nucleon(ro: wp.vec3, rd: wp.vec3, time: float, is_proton: int) -> wp.vec3:
    """Three color-charged quarks bound by gluon flux tubes.

    proton = up,up,down; neutron = up,down,down. Down quarks render dimmer.
    Color charges are red/green/blue -> the bound state is color-neutral.
    """
    conf = 1.0
    q0 = quark_pos(0, time, conf)
    q1 = quark_pos(1, time, conf)
    q2 = quark_pos(2, time, conf)

    # Flavor brightness: proton uud, neutron udd (down quarks dimmer).
    b1 = 0.7
    if is_proton == 1:
        b1 = 1.0

    qsize = 0.26
    col = color_charge(0) * emitter(ro, rd, q0, qsize)
    col = col + color_charge(1) * (emitter(ro, rd, q1, qsize) * b1)
    col = col + color_charge(2) * (emitter(ro, rd, q2, qsize) * 0.7)
    col = col * 1.7

    # Gluon flux tubes on all three quark pairs.
    g = flux(ro, rd, q0, q1, time) + flux(ro, rd, q1, q2, time) + flux(ro, rd, q2, q0, time)
    col = col + wp.vec3(0.8, 0.95, 1.0) * (g * 1.3)

    # Confinement 'bag' halo — warm for proton (+1), cool for neutron (0).
    bag = emitter(ro, rd, wp.vec3(0.0, 0.0, 0.0), conf * 0.85) * 0.06
    tint = wp.vec3(1.0, 0.9, 0.75)
    if is_proton == 0:
        tint = wp.vec3(0.8, 0.9, 1.0)
    return col + tint * bag
