"""Binary black hole — two holes spiralling into a merger, ray-traced through the field of both.

The hero of the relativity set. Two Schwarzschild black holes orbit their common centre; every
camera ray is a photon integrated through the **superposed** gravity of *both* — near either hole
it feels that hole's exact null-geodesic deflection (``a_k = -3/2 · h_k² · x_k / r_k⁵`` about
centre ``k``), and in between it feels the sum. So the two black shadows each lens the deep-sky
background *and each other*: look closely and each hole carries a warped little copy of its
companion and a bright **Einstein-ring** of photons that grazed its photon sphere, the "eyeholes"
signature of binary-black-hole imaging (Bohn et al. 2015).

Over ``--frames`` the pair **inspirals**: the separation shrinks and the orbit whirls faster
(the runaway chirp of a merger), and a quadrupole **gravitational-wave** shear ripples the lensed
starfield outward — the spacetime distortion LIGO heard on 14 Sept 2015, made visible. Render an
inspiral GIF and watch two shadows wheel together into one. See
``docs/research/43-relativistic-masterpieces.md``.
"""

import math

import warp as wp

from ..engine import post
from ..engine.blackhole import cosmic_background
from ..engine.color import kelvin_to_rgb
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_R_ESC = wp.constant(60.0)
_MAXSTEP = 900


@wp.func
def _pull(pos: wp.vec3, vel: wp.vec3, c: wp.vec3) -> wp.vec3:
    """Single-centre null-geodesic deflection about a hole at ``c`` (Schwarzschild, r_s = 1)."""
    x = pos - c
    r = wp.length(x)
    cr = wp.cross(x, vel)
    h2 = wp.dot(cr, cr)                              # angular momentum² about THIS centre
    return x * (-1.5 * h2 / (r * r * r * r * r))


@wp.func
def _ring_glow(r: float) -> float:
    """Hot photon-ring halo: brightest for rays grazing the photon sphere (r ≈ 1.5)."""
    d = (r - 1.5) / 0.14
    return wp.exp(-d * d)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   ca: wp.vec3, cb: wp.vec3, gw: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanf * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanf
    rd = wp.normalize(fwd + right * u + up * v)

    pos = eye
    vel = rd
    captured = int(0)
    glow = float(0.0)

    for _s in range(_MAXSTEP):
        ra = wp.length(pos - ca)
        rb = wp.length(pos - cb)
        if ra < 1.02 or rb < 1.02:
            captured = 1
            break
        if wp.length(pos) > _R_ESC:
            break
        acc = _pull(pos, vel, ca) + _pull(pos, vel, cb)
        dt = wp.clamp(0.14 * wp.min(ra, rb) / 3.0, 0.016, 0.36)
        vel = vel + acc * dt
        pos = pos + vel * dt
        glow = wp.min(glow + (_ring_glow(ra) + _ring_glow(rb)) * dt, 2.5)

    if captured == 1:
        col = wp.vec3(0.0, 0.0, 0.0)
    else:
        # gravitational-wave quadrupole shear on the escaped direction (grows near merger)
        d = wp.normalize(vel)
        shear = gw * (d[0] * d[0] - d[2] * d[2])
        d = wp.normalize(wp.vec3(d[0] * (1.0 + shear), d[1], d[2] * (1.0 - shear)))
        col = cosmic_background(d, 0.6)

    # photon-ring haloes: hot blue-white light bent around each hole (added even for captured rays
    # up to the horizon, so the shadows are rimmed with brilliant Einstein rings)
    col = col + kelvin_to_rgb(9000.0) * (glow * 0.16)
    img[i, j] = col


def _render(width, height, time, mouse, device):
    prog = min(float(time) * 0.16, 1.0)                     # 0 → 1 inspiral progress
    sep = 6.0 * (1.0 - prog) + 2.05 * prog                  # separation shrinks
    phase = float(time) * 0.9 + prog * prog * 3.4           # orbit whirls faster near merger
    phase = phase + float(mouse[0]) * 0.006
    c = sep * 0.5
    ca = wp.vec3(c * math.cos(phase), 0.0, c * math.sin(phase))
    cb = wp.vec3(-c * math.cos(phase), 0.0, -c * math.sin(phase))
    gw = 0.18 * prog * prog                                 # GW shear ramps into merger

    az = 0.35 + float(mouse[1]) * 0.004
    eye = wp.vec3(15.0 * math.sin(az), 6.5, -15.0 * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(40.0), ca, cb, gw],
              device=device)
    wp.synchronize_device(device)
    return post.tonemap(img.numpy(), mode="aces", exposure=1.15, preserve_hue=True)


SCENE = Scene(
    name="binary_bh",
    description="two Schwarzschild black holes spiralling into a merger, ray-traced through the "
                "superposed gravity of both — each shadow lenses the starfield and its companion "
                "('eyeholes'), rimmed by a bright Einstein photon-ring, while a quadrupole "
                "gravitational-wave shear ripples the lensed sky as the pair whirls together.",
    renderer=_render,
)
