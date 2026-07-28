"""Confinement, drawn — the Wilson-loop string breaking on a nucleated horizon.

Why can't you pull a single quark out of a proton? Holography's answer is a picture: a
quark-antiquark pair on the boundary is joined by a **string hanging into the bulk**
(Maldacena; Rey & Yee, 1998), and the interquark potential is the string's regularized
length. In AdS₃ the static string on a time slice minimizes proper length — it is the SAME
curve as the RT geodesic (``rt_geodesic_glow``'s orthogonal circle, here in 3D): one arc,
two entries of the dictionary.

The scene runs the two Hawking-Page phases of ``ads_hawking_page`` with a quark pair whose
separation slowly sweeps, and the string responds to the *ensemble*:

* **Confined (T < T_HP, thermal AdS).** No horizon exists. However far the quarks are
  pulled apart, the connected string survives — at wide separation it dives past the
  centre of the box, a single unbroken flux tube. Energy flows along it (the pulsing);
  separating the pair just makes more string. That IS confinement.
* **Deconfined (T > T_HP, the large hole).** The horizon offers the string a way out.
  The connected arc's deepest point is exact geometry (``string_turning_radius``:
  ``r_min = R(1 − sin α)/cos α``); once separation exceeds the **screening angle**
  (``screening_angle``: ``sin α = (R² − r_h²)/(R² + r_h²)``) the arc would dip inside the
  horizon — so it snaps into two radial strings falling into the hole. Each quark is now
  screened by the plasma: free at last, and the potential goes flat. Close pairs stay
  bound even in the plasma (quarkonium surviving deconfinement) — the break happens at
  the computed angle, not at the phase boundary.

The temperature cycle (~8 s) and the separation sweep (~20 s) run incommensurately, so the
string breaks and re-forms in different ways each cycle. Null geodesics, the reflecting
boundary, the nucleating hole and its locked Hawking temperature are all the
``ads_hawking_page`` machinery. See ``docs/research/46-ads-cft-holography.md`` (Part VI).
--frames runs the cycles; iMouse orbits.
"""

import math

import warp as wp

from .. import lod
from ..engine import post
from ..engine.adscft import (
    boundary_cft,
    hawking_page_temperature,
    horizon_radius,
    large_hole_radius,
    mass_of_radius,
    screening_angle,
    string_turning_radius,
)
from ..engine.pathtrace import camera_basis, tanfov
from ..scene import Scene

_L_ADS = 2.2
_R_BDY = wp.constant(24.0)
_OMEGA_T = 1.24                           # temperature cycle (the Hawking-Page dial)
_OMEGA_Q = 0.31                           # quark-separation sweep (4 temp cycles per sweep)
_BOUNCES = {"low": 1, "medium": 2, "high": 3, "ultra": 4}


@wp.func
def _string_emission(pos: wp.vec3, q1: wp.vec3, q2: wp.vec3, c3: wp.vec3, rho: float,
                     npl: wp.vec3, e1: wp.vec3, e2: wp.vec3, brk: float, r_cap: float,
                     time: float) -> wp.vec3:
    """Emission of the quark string sampled at a bulk point (both configurations).

    Connected: the circle through q1,q2 orthogonal to the boundary sphere (centre c3,
    radius rho, plane normal npl) — inside the ball that circle IS the geodesic string.
    Broken: two radial strings from each quark falling into the horizon at r_cap.
    The two are cross-faded by ``brk`` (the snap at the screening angle).
    """
    w = 0.22                               # string core width (box units)
    emit = wp.vec3(0.0, 0.0, 0.0)

    if brk < 0.999:                        # ---- connected flux tube ----
        v = pos - c3
        wn = wp.dot(v, npl)
        ip = v - npl * wn
        len2d = wp.length(ip)
        darc = wp.sqrt((len2d - rho) * (len2d - rho) + wn * wn)
        ang = wp.atan2(wp.dot(v, e2), wp.dot(v, e1))
        pulse = 0.72 + 0.28 * wp.sin(9.0 * ang - 3.2 * time)
        g = wp.exp(-(darc * darc) / (w * w))
        emit = emit + (wp.vec3(1.00, 0.78, 0.42) * g + wp.vec3(1.0, 1.0, 1.0) * (g * g)) \
            * (pulse * (1.0 - brk))

    if brk > 0.001:                        # ---- two broken strings draining into the hole ----
        for k in range(2):
            q = q1
            if k == 1:
                q = q2
            qn = q / wp.length(q)
            s = wp.clamp(wp.dot(pos, qn), r_cap, wp.length(q))
            dseg = wp.length(pos - qn * s)
            pulse = 0.72 + 0.28 * wp.sin(1.6 * s + 4.0 * time)   # energy falling inward
            g = wp.exp(-(dseg * dseg) / (w * w))
            emit = emit + (wp.vec3(1.00, 0.55, 0.30) * g + wp.vec3(1.0, 0.9, 0.8) * (g * g)) \
                * (pulse * brk)

    return emit


@wp.kernel
def _render_kernel(img: wp.array2d(dtype=wp.vec3), eye: wp.vec3, fwd: wp.vec3,
                   right: wp.vec3, up: wp.vec3, width: int, height: int, tanf: float,
                   time: float, max_steps: int, max_bounce: int,
                   m_bh: float, r_cap: float, t_bdy: float, hole: float,
                   q1: wp.vec3, q2: wp.vec3, c3: wp.vec3, rho: float, npl: wp.vec3,
                   e1: wp.vec3, e2: wp.vec3, brk: float):
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
    gas = float(0.0)

    for _s in range(max_steps):
        r = wp.length(pos)
        if hole > 0.0 and r < r_cap:
            break

        if r > _R_BDY:
            n = pos / r
            col = col + boundary_cft(n, time, t_bdy * 3.0) * (trans * 0.38)
            # the quarks live ON the boundary: hot endpoints where the string is anchored
            dq = wp.min(wp.length(pos - q1), wp.length(pos - q2))
            col = col + wp.vec3(1.0, 0.95, 0.85) * (wp.exp(-(dq * dq) / 1.4) * 2.2 * trans)
            bounce += 1
            if bounce > max_bounce:
                break
            trans = trans * 0.42
            vel = vel - n * (2.0 * wp.dot(vel, n))
            pos = n * (_R_BDY - 1.0e-3)
            cr2 = wp.cross(pos, vel)
            h2 = wp.dot(cr2, cr2)

        # exact Schwarzschild-AdS photon pull (Lambda drops out of the path shape)
        acc = pos * (-3.0 * m_bh * h2 / (r * r * r * r * r))
        dt = wp.clamp(0.16 * r / 3.0, 0.016, 0.5)
        vel = vel + acc * dt
        pos = pos + vel * dt

        # the string, sampled along the (bent) ray — lensed with everything else
        col = col + _string_emission(pos, q1, q2, c3, rho, npl, e1, e2, brk, r_cap, time) \
            * (trans * dt * 1.6)
        gas += dt * wp.exp(-r * 0.22) * (1.0 - hole)
        if hole > 0.0:
            gas += dt * wp.exp(-(r - r_cap) * 0.8) * hole * 0.25

    col = col + wp.vec3(1.00, 0.55, 0.25) * gas * t_bdy * 0.30
    img[i, j] = col


def _render(width, height, time, mouse, device):
    tier = lod.active_tier()
    max_steps = tier.raymarch_steps * 4
    max_bounce = _BOUNCES.get(tier.name, 2)
    t = float(time)

    # ---- the Hawking-Page dial (shared machinery with ads_hawking_page) ----
    t_hp = hawking_page_temperature(_L_ADS)
    t_bdy = t_hp * (1.0 + 0.05 * math.sin(_OMEGA_T * t))
    hole = min(max((t_bdy / t_hp - 1.0) / 0.02, 0.0), 1.0)
    if hole > 0.0:
        r_h = large_hole_radius(t_bdy, _L_ADS)
        m_bh = hole * mass_of_radius(r_h, _L_ADS)
        r_cap = horizon_radius(m_bh, _L_ADS) * 1.02 if m_bh > 0.0 else 0.0
    else:
        m_bh, r_cap = 0.0, 0.0

    # ---- the quark pair: separation sweeps; the string knows the exact geometry ----
    r_q = float(_R_BDY) - 2.0e-3
    dth = 1.70 + 1.35 * math.sin(_OMEGA_Q * t)          # quark separation (radians)
    alpha = 0.5 * dth
    qa = 0.9                                            # pair orientation in the equator
    q1 = wp.vec3(r_q * math.cos(qa + alpha), 0.0, r_q * math.sin(qa + alpha))
    q2 = wp.vec3(r_q * math.cos(qa - alpha), 0.0, r_q * math.sin(qa - alpha))
    # orthogonal circle through q1,q2: centre R/cos(a) along the bisector, radius R tan(a)
    e1 = wp.vec3(math.cos(qa), 0.0, math.sin(qa))
    npl = wp.vec3(0.0, 1.0, 0.0)
    e2 = wp.vec3(-math.sin(qa), 0.0, math.cos(qa))
    c3 = wp.vec3(e1[0] * r_q / math.cos(alpha), 0.0, e1[2] * r_q / math.cos(alpha))
    rho = r_q * math.tan(alpha)

    # ---- the snap: only a horizon can break the string, at the exact screening angle ----
    if hole > 0.0 and r_cap > 0.0:
        th_scr = screening_angle(r_cap, r_q)
        brk = hole * min(max((dth - th_scr) / 0.12 + 1.0, 0.0), 1.0)
        _ = string_turning_radius(dth, r_q)             # r_min < r_h exactly when dth > th_scr
    else:
        brk = 0.0

    az = t * 0.30 + float(mouse[0]) * 0.006
    dist = 22.0
    eye = wp.vec3(dist * math.sin(az), 2.4, -dist * math.cos(az))
    fwd, right, up = camera_basis(eye, wp.vec3(0.0, 0.0, 0.0))

    img = wp.zeros((height, width), dtype=wp.vec3, device=device)
    wp.launch(_render_kernel, dim=(height, width),
              inputs=[img, eye, fwd, right, up, width, height, tanfov(75.0), t,
                      max_steps, max_bounce, float(m_bh), float(r_cap), float(t_bdy),
                      float(hole), q1, q2, c3, float(rho), npl, e1, e2, float(brk)],
              device=device)
    wp.synchronize_device(device)
    hdr = img.numpy()
    hdr = post.bloom(hdr, threshold=0.9, strength=0.4, radius=5)
    return post.tonemap(hdr, mode="aces", exposure=1.1, preserve_hue=True)


SCENE = Scene(
    name="ads_confinement",
    description="confinement drawn holographically — a quark pair on the boundary joined by "
                "a Wilson-loop string hanging into the bulk (in AdS3 the same orthogonal-"
                "circle geodesic as the RT surface): below T_HP the connected flux tube "
                "survives any separation (confinement); above it the nucleated horizon snaps "
                "the string at the exact screening angle sin(a) = (R^2-r_h^2)/(R^2+r_h^2) "
                "into two strings draining into the hole — deconfined screening, with close "
                "pairs still bound (quarkonium). --frames runs both cycles.",
    renderer=_render,
)
