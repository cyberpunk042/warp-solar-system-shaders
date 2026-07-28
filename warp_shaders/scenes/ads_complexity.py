"""Holographic complexity — the wormhole grows forever.

The sharpest disconnect in the dictionary. Throw the eternal AdS black hole's dual CFT
out of equilibrium and it **thermalizes in a few thermal times**: correlators decay like
``e^{−2πt/β}``, entropies saturate, and every conventional boundary observable goes
static. But behind the horizon the Einstein-Rosen bridge **keeps growing** — the maximal
interior slice lengthens linearly in time, and keeps doing so for times exponential in
the entropy. What boundary quantity is still evolving when everything else has stopped?

Susskind's answer (2014): **computational complexity** — the minimum number of gates
needed to build the state — with the dictionary *complexity = interior volume* (CV), and
its refinement *complexity = action* (CA: the Wheeler-DeWitt patch action). CA's
late-time growth rate is exactly ``dC/dt = 2M/π`` — **the Lloyd bound**: black holes are
nature's fastest computers, computing as fast as physics allows
(Brown-Roberts-Susskind-Swingle-Zhao 2015). The engine carries the dictionary host-side:
``lloyd_bound`` (2M/π), ``complexity_rate`` (tanh ramp approaching the bound from below,
never exceeding it — test-asserted), ``complexity_growth`` (quadratic early, linear
forever after), ``scrambling_time`` (t_* = (β/2π)·ln S, the fast-scrambler time).

The scene stages the disconnect directly, one cycle = one thermalization:

* The boundary lattice starts hot and flickering (the quench) and **freezes** over the
  first couple of seconds — thermal equilibrium; nothing outside changes anymore.
* Inside the shadow, the interior: a violet-white **pillar of light** — the maximal
  slice through the Einstein-Rosen bridge — whose height tracks ``complexity_growth``
  and whose brightness tracks ``complexity_rate`` saturating the Lloyd bound. It keeps
  growing long after the boundary has gone static: the only thing in the picture still
  computing.

The exterior geodesics, the reflecting boundary, and the growth/bound closed forms are
the honest dictionary; the pillar is the interior volume drawn schematically inside the
capture region. See ``docs/research/46-ads-cft-holography.md`` (Part VIII). --frames runs
one thermalization cycle; iMouse orbits.
"""

import math

import warp as wp

from .. import lod
from ..engine import post
from ..engine.adscft import (
    boundary_cft,
    complexity_growth,
    complexity_rate,
    hawking_temperature,
    lloyd_bound,
)
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_M_BH = 0.5
_L_ADS = 7.0
_R_BDY = wp.constant(14.0)
_T_HAWK = hawking_temperature(_M_BH, _L_ADS)
_BOUNCES = {"low": 1, "medium": 2, "high": 3, "ultra": 4}
_T_CYCLE = 14.0                           # one quench-and-grow cycle (seconds)
_T_RAMP = 3.0                             # complexity-rate ramp time
_T_THERM = 1.2                            # boundary thermalization time (fast!)


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   t_anim: float, max_steps: int, max_bounce: int, t_hawk: float,
                   pillar_h: float, rate_n: float):
    i, j = wp.tid()
    aspect = float(width) / float(height)
    u = (2.0 * (float(j) + 0.5) / float(width) - 1.0) * tanf * aspect
    v = (2.0 * (float(height - 1 - i) + 0.5) / float(height) - 1.0) * tanf
    rd = wp.normalize(fwd + right * u + up * v)

    pos = eye
    vel = rd
    cr = wp.cross(pos, vel)
    h2 = wp.dot(cr, cr)

    col = wp.vec3(0.0, 0.0, 0.0)
    trans = float(1.0)
    bounce = int(0)

    for _s in range(max_steps):
        r = wp.length(pos)

        if r < 1.02:
            # captured: inside the shadow. The interior is not nothing — the maximal
            # slice through the Einstein-Rosen bridge, still growing: a pillar of light
            # whose height is C(t) and whose brightness is dC/dt vs the Lloyd bound.
            # March the ray straight on through the (schematic) interior and volume-
            # accumulate the pillar.
            for _k in range(60):
                pos = pos + vel * 0.05
                rho = wp.sqrt(pos[0] * pos[0] + pos[2] * pos[2])
                if wp.abs(pos[1]) < pillar_h and rho < 0.5:
                    glow = wp.exp(-rho * 2.0) * 0.05 * rate_n * 2.5 * trans
                    col = col + wp.vec3(0.82, 0.68, 1.00) * glow
            break

        if r > _R_BDY:
            n = pos / r
            col = col + boundary_cft(n, t_anim, t_hawk) * trans
            bounce += 1
            if bounce > max_bounce:
                break
            trans = trans * 0.42
            vel = vel - n * (2.0 * wp.dot(vel, n))
            pos = n * (_R_BDY - 1.0e-3)
            cr2 = wp.cross(pos, vel)
            h2 = wp.dot(cr2, cr2)

        # exact Schwarzschild-AdS photon pull (Lambda drops out of the path shape)
        acc = pos * (-1.5 * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.45)
        vel = vel + acc * dt
        pos = pos + vel * dt

    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = lod.active_tier()
    max_steps = tier.raymarch_steps * 4
    max_bounce = _BOUNCES.get(tier.name, 2)

    tau = math.fmod(float(time), _T_CYCLE)

    # ---- the boundary thermalizes FAST: lattice animation freezes into equilibrium ----
    # (the animation clock saturates: correlators e^{-2 pi t / beta} -> static thermal)
    t_anim = _T_THERM * (1.0 - math.exp(-tau / _T_THERM))

    # ---- the interior keeps growing: the dictionary, closed-form ----
    c_now = complexity_growth(tau, _M_BH, _T_RAMP)
    c_end = complexity_growth(_T_CYCLE, _M_BH, _T_RAMP)
    rate_n = complexity_rate(tau, _M_BH, _T_RAMP) / lloyd_bound(_M_BH)   # -> 1, never past
    pillar_h = 0.10 + 0.85 * (c_now / c_end)              # the maximal slice, lengthening

    az = float(time) * 0.30 + float(mouse[0]) * 0.006
    dist = 8.5
    eye = wp.vec3(dist * math.sin(az), 1.8, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(52.0), float(t_anim),
                      max_steps, max_bounce, float(_T_HAWK), float(pillar_h), float(rate_n)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ads_complexity",
    description="the wormhole grows forever — Susskind's complexity=volume staged as a "
                "disconnect: the boundary CFT thermalizes in seconds (the lattice freezes "
                "into static equilibrium) while inside the shadow the Einstein-Rosen "
                "interior keeps growing, drawn as a violet pillar of light whose height "
                "tracks C(t) = (2M/pi) t_ramp ln cosh(t/t_ramp) and whose brightness "
                "tracks dC/dt saturating the Lloyd bound 2M/pi from below — black holes "
                "as nature's fastest computers, still computing long after everything "
                "visible has stopped. --frames runs one thermalization cycle.",
    renderer=_render,
)
