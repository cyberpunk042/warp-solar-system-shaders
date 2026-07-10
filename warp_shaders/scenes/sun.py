"""Sun / star corona — an NVIDIA Warp port of trisomie21's GLSL shader
(https://www.shadertoy.com/view/lsf3RH). A turbulent star with a flaring corona.

The original is channel-driven:
  - ``iChannel1`` is an audio FFT texture — ``freqs[]`` samples drive brightness.
  - ``iChannel0`` is a granular star texture for the surface + star sphere.

Neither exists in an offline renderer, so this port follows the gallery
convention for channel-driven Shadertoy scenes:
  - audio  -> dropped; ``brightness`` is a fixed constant (no sound),
  - texture -> procedural fBm noise (``sun_tex``), so no image assets are needed.

The surface still animates over time through the noise term. Original GLSL is
kept at ``reference/sun.frag``.
"""

import warp as wp

from ..scene import Scene
from ..sdf import fract, noise2d


@wp.func
def _mod1(x: float, y: float) -> float:
    # GLSL mod(x, y)
    return x - y * wp.floor(x / y)


@wp.func
def snoise(ux: float, uy: float, uz: float, res: float) -> float:
    """3D value noise (trisomie21's ``snoise``), unrolled to scalars."""
    ux = ux * res
    uy = uy * res
    uz = uz * res

    u0x = wp.floor(_mod1(ux, res)) * 1.0
    u0y = wp.floor(_mod1(uy, res)) * 100.0
    u0z = wp.floor(_mod1(uz, res)) * 10000.0
    u1x = wp.floor(_mod1(ux + 1.0, res)) * 1.0
    u1y = wp.floor(_mod1(uy + 1.0, res)) * 100.0
    u1z = wp.floor(_mod1(uz + 1.0, res)) * 10000.0

    fx = fract(ux)
    fy = fract(uy)
    fz = fract(uz)
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    fz = fz * fz * (3.0 - 2.0 * fz)

    v0 = u0x + u0y + u0z
    v1 = u1x + u0y + u0z
    v2 = u0x + u1y + u0z
    v3 = u1x + u1y + u0z

    r0x = fract(wp.sin(v0 * 1.0e-3) * 1.0e5)
    r0y = fract(wp.sin(v1 * 1.0e-3) * 1.0e5)
    r0z = fract(wp.sin(v2 * 1.0e-3) * 1.0e5)
    r0w = fract(wp.sin(v3 * 1.0e-3) * 1.0e5)
    r0 = wp.lerp(wp.lerp(r0x, r0y, fx), wp.lerp(r0z, r0w, fx), fy)

    off = u1z - u0z
    r1x = fract(wp.sin((v0 + off) * 1.0e-3) * 1.0e5)
    r1y = fract(wp.sin((v1 + off) * 1.0e-3) * 1.0e5)
    r1z = fract(wp.sin((v2 + off) * 1.0e-3) * 1.0e5)
    r1w = fract(wp.sin((v3 + off) * 1.0e-3) * 1.0e5)
    r1 = wp.lerp(wp.lerp(r1x, r1y, fx), wp.lerp(r1z, r1w, fx), fy)

    return wp.lerp(r0, r1, fz) * 2.0 - 1.0


@wp.func
def sun_tex(u: float, v: float) -> wp.vec3:
    """Procedural stand-in for iChannel0 — a granular, warm-tinted star texture."""
    val = float(0.0)
    amp = float(0.6)
    pu = u * 3.0
    pv = v * 3.0
    for _ in range(5):
        val += amp * noise2d(wp.vec2(pu, pv))
        pu *= 2.0
        pv *= 2.0
        amp *= 0.5
    val = wp.clamp(val, 0.0, 1.0)
    return wp.vec3(val, val * 0.85, val * 0.7)


@wp.kernel
def render_kernel(
    img: wp.array2d(dtype=wp.vec3),
    width: int,
    height: int,
    time: float,
    mouse: wp.vec2,
):
    i, j = wp.tid()

    # Original was audio-reactive (iChannel1 FFT -> brightness). No sound here:
    # a fixed brightness. The surface still churns via `time` in the noise below.
    brightness = float(0.3)

    radius = 0.24 + brightness * 0.2
    inv_radius = 1.0 / radius

    orange = wp.vec3(0.8, 0.65, 0.3)
    orange_red = wp.vec3(0.8, 0.35, 0.1)

    tt = time * 0.1  # GLSL `time = iTime * 0.1`
    res = wp.vec2(float(width), float(height))
    aspect = res[0] / res[1]

    uvx = float(j) / res[0]
    uvy = float(height - 1 - i) / res[1]  # GLSL uv is bottom-up

    px = (-0.5 + uvx) * aspect
    py = -0.5 + uvy

    fade = wp.sqrt(wp.length(wp.vec2(2.0 * px, 2.0 * py)))
    f_val1 = 1.0 - fade
    f_val2 = 1.0 - fade

    angle = wp.atan2(px, py) / 6.2832
    dist = wp.length(wp.vec2(px, py))

    cx = angle
    cy = dist
    cz = tt * 0.1

    new_time1 = wp.abs(snoise(cx, cy - tt * (0.35 + brightness * 0.001), cz + tt * 0.015, 15.0))
    new_time2 = wp.abs(snoise(cx, cy - tt * (0.15 + brightness * 0.001), cz + tt * 0.015, 45.0))

    for k in range(1, 8):
        power = wp.pow(2.0, float(k + 1))
        f_val1 += (0.5 / power) * snoise(cx, cy - tt, cz + tt * 0.2, power * 10.0 * (new_time1 + 1.0))
        f_val2 += (0.5 / power) * snoise(cx, cy - tt, cz + tt * 0.2, power * 25.0 * (new_time2 + 1.0))

    c1 = f_val1 * wp.max(1.1 - fade, 0.0)
    c2 = f_val2 * wp.max(1.1 - fade, 0.0)
    corona = (c1 * c1) * 50.0 + (c2 * c2) * 50.0
    corona *= 1.2 - new_time1

    spx = (-1.0 + 2.0 * uvx) * aspect * (2.0 - brightness)
    spy = (-1.0 + 2.0 * uvy) * (2.0 - brightness)
    r = spx * spx + spy * spy
    f = (1.0 - wp.sqrt(wp.abs(1.0 - r))) / (r + 1.0e-6) + brightness * 0.5

    star_sphere = wp.vec3(0.0, 0.0, 0.0)
    if dist < radius:
        corona *= wp.pow(dist * inv_radius, 24.0)
        nux = spx * f + tt
        nuy = spy * f
        tex_g = sun_tex(nux, nuy)[1]
        u_off = tex_g * brightness * 4.5 + tt
        star_sphere = sun_tex(nux + u_off, nuy)

    star_glow = wp.clamp(1.0 - dist * (1.0 - brightness), 0.0, 1.0)

    col = orange * (f * (0.75 + brightness * 0.3))
    col = col + star_sphere + orange * corona + orange_red * star_glow
    img[i, j] = col


SCENE = Scene(
    name="sun",
    kernel=render_kernel,
    description="Turbulent star with a flaring corona (trisomie21). Texture->procedural; no audio.",
)
